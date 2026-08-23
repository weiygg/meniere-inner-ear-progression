from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from skimage.measure import label, regionprops
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


STRUCT_ALIASES = {
    "COCHLEAR": "Cochlear",
    "CHOCHLEAR": "Cochlear",
    "CHOLEAR": "Cochlear",
    "VESTIBULAR": "Vestibular",
    "SSC": "SSC",
    "HSC": "HSC",
    "PSC": "PSC",
    "TV": "TV",
    "ELS": "ELS",
}
CANALS = ("SSC", "HSC", "PSC")
VOLUME_STRUCTS = ("ELS", "TV")
FNAME_PATTERN = re.compile(
    r"(?P<pid>\d+)(?P<side>[LR])[-_](?P<struct>[A-Za-z]+)\.nii(\.gz)?$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast angle/ELS-TV correlation analysis.")
    parser.add_argument("--data-dir", default="seg4")
    parser.add_argument("--output-dir", default="analysis_out/seg4_20260522/quick_angle_corr")
    parser.add_argument("--min-component-voxels", type=int, default=0)
    parser.add_argument("--max-plane-points", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def canonical_struct(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z]", "", name).upper()
    return STRUCT_ALIASES.get(cleaned, cleaned.title())


def collect_files(data_dir: Path) -> dict[tuple[str, str, str], Path]:
    files: dict[tuple[str, str, str], Path] = {}
    for path in data_dir.rglob("*.nii*"):
        match = FNAME_PATTERN.search(path.name)
        if not match:
            continue
        pid = match.group("pid").zfill(3)
        side = match.group("side").upper()
        struct = canonical_struct(match.group("struct"))
        if struct in set(CANALS) | set(VOLUME_STRUCTS):
            files.setdefault((pid, side, struct), path)
    return files


def load_mask(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = nib.load(str(path))
    mask = np.asanyarray(img.dataobj) > 0.5
    zooms = tuple(float(v) for v in img.header.get_zooms()[:3])
    return mask, zooms


def largest_component(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    if min_voxels <= 0:
        return mask
    if int(mask.sum()) < min_voxels:
        return mask
    lab = label(mask)
    if int(lab.max()) == 0:
        return mask
    props = sorted(regionprops(lab), key=lambda r: r.area, reverse=True)
    return lab == props[0].label


def volume_mm3(mask: np.ndarray, zooms: tuple[float, float, float]) -> float:
    return float(mask.sum() * zooms[0] * zooms[1] * zooms[2])


def fit_plane(mask: np.ndarray, zooms: tuple[float, float, float], max_points: int, rng: np.random.Generator) -> dict[str, float]:
    points = np.argwhere(mask)
    if points.shape[0] < 3:
        return {"ok": 0}
    if points.shape[0] > max_points:
        points = points[rng.choice(points.shape[0], size=max_points, replace=False)]
    points_mm = points.astype(float) * np.asarray(zooms, dtype=float)
    center = points_mm.mean(axis=0)
    x = points_mm - center
    _, svals, vt = np.linalg.svd(x, full_matrices=False)
    normal = vt[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    distances = np.abs(x @ normal)
    return {
        "ok": 1,
        "n_points": int(points_mm.shape[0]),
        "nx": float(normal[0]),
        "ny": float(normal[1]),
        "nz": float(normal[2]),
        "cx": float(center[0]),
        "cy": float(center[1]),
        "cz": float(center[2]),
        "plane_rmse_mm": float(np.sqrt(np.mean(distances**2))),
        "plane_mae_mm": float(np.mean(distances)),
        "sv1": float(svals[0]) if svals.shape[0] > 0 else math.nan,
        "sv2": float(svals[1]) if svals.shape[0] > 1 else math.nan,
        "sv3": float(svals[2]) if svals.shape[0] > 2 else math.nan,
    }


def angle_between(n1: np.ndarray, n2: np.ndarray) -> float:
    n1 = n1 / (np.linalg.norm(n1) + 1e-12)
    n2 = n2 / (np.linalg.norm(n2) + 1e-12)
    cos_value = float(np.clip(abs(np.dot(n1, n2)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_value)))


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 and np.std(y_pred) > 0 else math.nan,
    }


def correlation_table(df: pd.DataFrame, targets: list[str], features: list[str]) -> pd.DataFrame:
    rows = []
    for target in targets:
        for feature in features:
            valid = df[[target, feature]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) < 3:
                continue
            pr, pp = pearsonr(valid[feature], valid[target])
            sr, sp = spearmanr(valid[feature], valid[target])
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "n": int(len(valid)),
                    "pearson_r": float(pr),
                    "pearson_p": float(pp),
                    "spearman_r": float(sr),
                    "spearman_p": float(sp),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_pearson_r"] = out["pearson_r"].abs()
        out["abs_spearman_r"] = out["spearman_r"].abs()
        out = out.sort_values(["target", "abs_pearson_r"], ascending=[True, False])
    return out


def model_table(df: pd.DataFrame, target: str, features: list[str], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = df[["pid", "side", target, *features]].replace([np.inf, -np.inf], np.nan).dropna(subset=[target]).copy()
    x = valid[features]
    y = valid[target].to_numpy(dtype=float)
    cv = LeaveOneOut() if len(valid) <= 80 else KFold(n_splits=5, shuffle=True, random_state=seed)
    cv_name = "leave_one_out" if len(valid) <= 80 else "kfold_5"
    models = {
        "baseline_mean": DummyRegressor(strategy="mean"),
        "ridge_angle_only": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
            ]
        ),
        "random_forest_angle_only": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=500, min_samples_leaf=4, random_state=seed)),
            ]
        ),
    }
    metric_rows = []
    pred_frames = []
    for name, model in models.items():
        pred = cross_val_predict(model, x, y, cv=cv, n_jobs=1)
        metric_rows.append({"target": target, "model": name, "cv": cv_name, "n": int(len(valid)), **score(y, pred)})
        pred_frames.append(pd.DataFrame({"pid": valid["pid"], "side": valid["side"], "target": target, "model": name, "y_true": y, "y_pred": pred}))
    metrics = pd.DataFrame(metric_rows).sort_values(["rmse", "mae", "r2"], ascending=[True, True, False])
    predictions = pd.concat(pred_frames, ignore_index=True)
    predictions["residual"] = predictions["y_true"] - predictions["y_pred"]
    return metrics, predictions


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    files = collect_files(data_dir)
    grouped: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)
    for (pid, side, struct), path in files.items():
        grouped[(pid, side)][struct] = path

    volume_rows = []
    plane_rows = []
    ear_rows = []
    required = set(CANALS) | set(VOLUME_STRUCTS)
    complete_groups = [(key, by_struct) for key, by_struct in grouped.items() if required.issubset(by_struct)]

    for (pid, side), by_struct in sorted(complete_groups):
        row = {"pid": pid, "side": side}
        normals: dict[str, np.ndarray] = {}
        for struct in VOLUME_STRUCTS:
            if struct not in by_struct:
                continue
            mask, zooms = load_mask(by_struct[struct])
            mask = largest_component(mask, args.min_component_voxels)
            vol = volume_mm3(mask, zooms)
            row[f"{struct.lower()}_mm3"] = vol
            volume_rows.append({"pid": pid, "side": side, "struct": struct, "volume_mm3": vol, "file": str(by_struct[struct])})
        for struct in CANALS:
            if struct not in by_struct:
                continue
            mask, zooms = load_mask(by_struct[struct])
            mask = largest_component(mask, args.min_component_voxels)
            plane = fit_plane(mask, zooms, args.max_plane_points, rng)
            plane_rows.append({"pid": pid, "side": side, "struct": struct, **plane, "file": str(by_struct[struct])})
            if plane["ok"]:
                normals[struct] = np.array([plane["nx"], plane["ny"], plane["nz"]], dtype=float)
                for key, value in plane.items():
                    row[f"{struct.lower()}_{key}"] = value
        row["has_els_tv"] = int("els_mm3" in row and "tv_mm3" in row and row.get("tv_mm3", 0) > 0)
        row["complete_three_canals"] = int(all(struct in normals for struct in CANALS))
        if row["has_els_tv"]:
            row["els_over_tv"] = row["els_mm3"] / row["tv_mm3"]
            row["els_over_tv_plus_els"] = row["els_mm3"] / (row["els_mm3"] + row["tv_mm3"])
        if row["complete_three_canals"]:
            row["angle_ssc_hsc_deg"] = angle_between(normals["SSC"], normals["HSC"])
            row["angle_ssc_psc_deg"] = angle_between(normals["SSC"], normals["PSC"])
            row["angle_hsc_psc_deg"] = angle_between(normals["HSC"], normals["PSC"])
            row["orthogonality_deviation_mean_deg"] = float(
                np.mean(
                    [
                        abs(row["angle_ssc_hsc_deg"] - 90),
                        abs(row["angle_ssc_psc_deg"] - 90),
                        abs(row["angle_hsc_psc_deg"] - 90),
                    ]
                )
            )
        ear_rows.append(row)

    volumes = pd.DataFrame(volume_rows)
    planes = pd.DataFrame(plane_rows)
    ears = pd.DataFrame(ear_rows)
    volumes.to_csv(output_dir / "volumes_mm3_quick.csv", index=False, encoding="utf-8-sig")
    planes.to_csv(output_dir / "plane_features_quick.csv", index=False, encoding="utf-8-sig")
    ears.to_csv(output_dir / "angle_dataset.csv", index=False, encoding="utf-8-sig")

    angle_features = [
        "angle_ssc_hsc_deg",
        "angle_ssc_psc_deg",
        "angle_hsc_psc_deg",
        "orthogonality_deviation_mean_deg",
    ]
    targets = ["els_over_tv", "els_over_tv_plus_els", "els_mm3", "tv_mm3"]
    analysis_df = ears[(ears["has_els_tv"] == 1) & (ears["complete_three_canals"] == 1)].copy()
    corr = correlation_table(analysis_df, targets, angle_features)
    corr.to_csv(output_dir / "angle_correlations.csv", index=False, encoding="utf-8-sig")

    all_metrics = []
    all_predictions = []
    for target in targets:
        metrics, predictions = model_table(analysis_df, target, angle_features, args.seed)
        all_metrics.append(metrics)
        all_predictions.append(predictions)
    model_metrics = pd.concat(all_metrics, ignore_index=True)
    cv_predictions = pd.concat(all_predictions, ignore_index=True)
    model_metrics.to_csv(output_dir / "angle_only_model_comparison.csv", index=False, encoding="utf-8-sig")
    cv_predictions.to_csv(output_dir / "angle_only_cv_predictions.csv", index=False, encoding="utf-8-sig")

    summary = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "n_subject_dirs": len([p for p in data_dir.iterdir() if p.is_dir()]),
        "n_ears_total": int(len(ears)),
        "n_ears_with_els_tv": int(ears["has_els_tv"].sum()) if "has_els_tv" in ears else 0,
        "n_ears_with_els_tv_and_three_canals": int(len(analysis_df)),
        "files": {
            "angle_dataset": str(output_dir / "angle_dataset.csv"),
            "angle_correlations": str(output_dir / "angle_correlations.csv"),
            "angle_only_model_comparison": str(output_dir / "angle_only_model_comparison.csv"),
            "angle_only_cv_predictions": str(output_dir / "angle_only_cv_predictions.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
