from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import numpy as np


ANGLE_TYPES = ("SSC-HSC", "SSC-PSC", "HSC-PSC")
RAW_ANGLE_TO_FEATURE = {
    "SSC-HSC": "angle_ssc_hsc_deg",
    "SSC-PSC": "angle_ssc_psc_deg",
    "HSC-PSC": "angle_hsc_psc_deg",
}
FEATURE_COLS = [
    "angle_ssc_hsc_deg",
    "angle_ssc_psc_deg",
    "angle_hsc_psc_deg",
    "dev_ssc_hsc_from_90",
    "dev_ssc_psc_from_90",
    "dev_hsc_psc_from_90",
    "angle_mean_deg",
    "angle_std_deg",
    "orthogonality_deviation_mean_deg",
    "side_is_right",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model inner-ear ELS volume ratio from semicircular canal angles."
    )
    parser.add_argument(
        "--analysis-dir",
        default=None,
        help="Folder containing canal_plane_angles_deg.csv and volumes_mm3.csv. Default: auto-select best analysis_out folder in cwd.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: <analysis-dir>/angle_ratio_model",
    )
    parser.add_argument(
        "--target",
        choices=("els_over_tv", "els_over_tv_plus_els", "els_mm3"),
        default="els_over_tv",
        help="Regression target.",
    )
    parser.add_argument(
        "--predict-angle-csv",
        default=None,
        help="Optional angle CSV for inference. Accepts raw canal_plane_angles_deg.csv or a prebuilt feature CSV.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Saved model bundle (.pkl). Required for inference-only mode; optional for training mode.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def zfill_pid(value: str) -> str:
    text = str(value).strip()
    return text.zfill(3) if text.isdigit() else text


def infer_best_analysis_dir(user_value: str | None) -> Path:
    if user_value:
        path = Path(user_value)
        if not path.exists():
            raise FileNotFoundError(f"Analysis directory not found: {path}")
        return path

    candidates: list[tuple[int, int, str, Path]] = []
    for child in Path.cwd().iterdir():
        if not child.is_dir():
            continue
        analysis_dir = child / "analysis_out"
        angle_csv = analysis_dir / "canal_plane_angles_deg.csv"
        volume_csv = analysis_dir / "volumes_mm3.csv"
        if not angle_csv.exists() or not volume_csv.exists():
            continue
        feature_rows = load_angle_features(angle_csv)
        target_rows, _ = load_target_rows(volume_csv, "els_over_tv")
        overlap = len(set(feature_rows) & set(target_rows))
        candidates.append((overlap, len(feature_rows), child.name, analysis_dir))

    if not candidates:
        raise FileNotFoundError("No usable analysis_out folder was found in the current directory.")

    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return candidates[0][3]


def prepare_output_dir(user_value: str | None, analysis_dir: Path) -> Path:
    out_dir = Path(user_value) if user_value else analysis_dir / "angle_ratio_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_angle_features(angle_csv: Path) -> dict[tuple[str, str], dict[str, float]]:
    rows = read_csv_rows(angle_csv)
    if not rows:
        return {}

    columns = set(rows[0].keys())
    feature_map: dict[tuple[str, str], dict[str, float]] = {}

    if {"pid", "side", "angle_type", "angle_deg"} <= columns:
        raw_map: dict[tuple[str, str], dict[str, float]] = {}
        for row in rows:
            key = (zfill_pid(row["pid"]), row["side"].strip().upper())
            raw_map.setdefault(key, {})
            raw_map[key][row["angle_type"].strip()] = float(row["angle_deg"])
        for key, raw in raw_map.items():
            if not all(angle_type in raw for angle_type in ANGLE_TYPES):
                continue
            feature_map[key] = derive_angle_features(raw, key[1])
        return feature_map

    required = {"pid", "side", *RAW_ANGLE_TO_FEATURE.values()}
    missing = required - columns
    if missing:
        raise ValueError(f"Prediction feature CSV is missing columns: {sorted(missing)}")

    for row in rows:
        key = (zfill_pid(row["pid"]), row["side"].strip().upper())
        raw = {
            "SSC-HSC": float(row["angle_ssc_hsc_deg"]),
            "SSC-PSC": float(row["angle_ssc_psc_deg"]),
            "HSC-PSC": float(row["angle_hsc_psc_deg"]),
        }
        feature_map[key] = derive_angle_features(raw, key[1])
    return feature_map


def derive_angle_features(raw_angles: dict[str, float], side: str) -> dict[str, float]:
    feat = {
        "angle_ssc_hsc_deg": float(raw_angles["SSC-HSC"]),
        "angle_ssc_psc_deg": float(raw_angles["SSC-PSC"]),
        "angle_hsc_psc_deg": float(raw_angles["HSC-PSC"]),
    }
    feat["dev_ssc_hsc_from_90"] = abs(feat["angle_ssc_hsc_deg"] - 90.0)
    feat["dev_ssc_psc_from_90"] = abs(feat["angle_ssc_psc_deg"] - 90.0)
    feat["dev_hsc_psc_from_90"] = abs(feat["angle_hsc_psc_deg"] - 90.0)
    raw_values = np.array(
        [feat["angle_ssc_hsc_deg"], feat["angle_ssc_psc_deg"], feat["angle_hsc_psc_deg"]],
        dtype=float,
    )
    dev_values = np.array(
        [feat["dev_ssc_hsc_from_90"], feat["dev_ssc_psc_from_90"], feat["dev_hsc_psc_from_90"]],
        dtype=float,
    )
    feat["angle_mean_deg"] = float(np.mean(raw_values))
    feat["angle_std_deg"] = float(np.std(raw_values))
    feat["orthogonality_deviation_mean_deg"] = float(np.mean(dev_values))
    feat["side_is_right"] = 1.0 if side.upper() == "R" else 0.0
    return feat


def load_target_rows(volume_csv: Path, target: str) -> tuple[dict[tuple[str, str], float], list[str]]:
    rows = read_csv_rows(volume_csv)
    volume_map: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        key = (zfill_pid(row["pid"]), row["side"].strip().upper())
        volume_map.setdefault(key, {})
        volume_map[key][row["struct"].strip()] = float(row["volume_mm3"])

    target_map: dict[tuple[str, str], float] = {}
    for key, structs in volume_map.items():
        if "ELS" not in structs:
            continue
        els = float(structs["ELS"])
        if target == "els_mm3":
            target_map[key] = els
            continue
        if "TV" not in structs:
            continue
        tv = float(structs["TV"])
        if tv <= 0:
            continue
        if target == "els_over_tv":
            target_map[key] = els / tv
        elif target == "els_over_tv_plus_els":
            denom = els + tv
            if denom <= 0:
                continue
            target_map[key] = els / denom
        else:
            raise ValueError(f"Unsupported target: {target}")
    return target_map, ["ELS", "TV"]


def build_training_rows(analysis_dir: Path, target: str) -> list[dict]:
    feature_map = load_angle_features(analysis_dir / "canal_plane_angles_deg.csv")
    target_map, _ = load_target_rows(analysis_dir / "volumes_mm3.csv", target)

    rows: list[dict] = []
    for key in sorted(set(feature_map) & set(target_map)):
        pid, side = key
        row = {"pid": pid, "side": side, **feature_map[key], "target_value": target_map[key]}
        rows.append(row)
    if not rows:
        raise RuntimeError("No overlapping ears were found between angle features and target volumes.")
    return rows


def rows_to_arrays(rows: list[dict], feature_cols: list[str], target_col: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[float(row[col]) for col in feature_cols] for row in rows], dtype=float)
    y = np.array([float(row[target_col]) for row in rows], dtype=float)
    return x, y


def make_splits(num_rows: int, random_state: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    indices = np.arange(num_rows)
    if num_rows <= 80:
        return [(np.delete(indices, i), np.array([i])) for i in range(num_rows)], "leave_one_out"

    rng = np.random.default_rng(random_state)
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, 5)
    splits = []
    for test_idx in folds:
        train_idx = np.array(sorted(set(indices.tolist()) - set(test_idx.tolist())))
        splits.append((train_idx, np.array(sorted(test_idx.tolist()))))
    return splits, "kfold_5"


def make_inner_splits(num_rows: int, random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(num_rows)
    if num_rows <= 20:
        return [(np.delete(indices, i), np.array([i])) for i in range(num_rows)]
    rng = np.random.default_rng(random_state)
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, min(5, num_rows))
    splits = []
    for test_idx in folds:
        if test_idx.size == 0:
            continue
        train_idx = np.array(sorted(set(indices.tolist()) - set(test_idx.tolist())))
        splits.append((train_idx, np.array(sorted(test_idx.tolist()))))
    return splits


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def standardize_apply(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (x - mean) / scale


def fit_linear_model(x: np.ndarray, y: np.ndarray) -> dict:
    mean, scale = standardize_fit(x)
    xs = standardize_apply(x, mean, scale)
    design = np.concatenate([np.ones((xs.shape[0], 1), dtype=float), xs], axis=1)
    coef = np.linalg.pinv(design) @ y
    return {
        "type": "linear",
        "x_mean": mean.tolist(),
        "x_scale": scale.tolist(),
        "coef": coef.tolist(),
    }


def fit_ridge_fixed_alpha(x: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    mean, scale = standardize_fit(x)
    xs = standardize_apply(x, mean, scale)
    design = np.concatenate([np.ones((xs.shape[0], 1), dtype=float), xs], axis=1)
    penalty = np.eye(design.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + alpha * penalty, design.T @ y)
    return {
        "type": "ridge",
        "x_mean": mean.tolist(),
        "x_scale": scale.tolist(),
        "coef": coef.tolist(),
        "alpha": float(alpha),
    }


def predict_model(model: dict, x: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["x_mean"], dtype=float)
    scale = np.asarray(model["x_scale"], dtype=float)
    coef = np.asarray(model["coef"], dtype=float)
    xs = standardize_apply(x, mean, scale)
    design = np.concatenate([np.ones((xs.shape[0], 1), dtype=float), xs], axis=1)
    return design @ coef


def select_ridge_alpha(x: np.ndarray, y: np.ndarray, random_state: int) -> float:
    alphas = np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0], dtype=float)
    splits = make_inner_splits(len(y), random_state)
    best_alpha = float(alphas[0])
    best_rmse = float("inf")
    for alpha in alphas:
        preds = np.zeros_like(y, dtype=float)
        for train_idx, test_idx in splits:
            model = fit_ridge_fixed_alpha(x[train_idx], y[train_idx], float(alpha))
            preds[test_idx] = predict_model(model, x[test_idx])
        rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = float(alpha)
    return best_alpha


def fit_ridge_model(x: np.ndarray, y: np.ndarray, random_state: int) -> dict:
    alpha = select_ridge_alpha(x, y, random_state)
    return fit_ridge_fixed_alpha(x, y, alpha)


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    y_true_centered = y_true - np.mean(y_true)
    y_pred_centered = y_pred - np.mean(y_pred)
    denom = np.linalg.norm(y_true_centered) * np.linalg.norm(y_pred_centered)
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(y_true_centered, y_pred_centered) / denom)


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    if np.var(y_true) < 1e-12:
        r2 = float("nan")
    else:
        r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "mae": mae,
        "rmse": float(math.sqrt(mse)),
        "r2": r2,
        "pearson_r": pearson_r(y_true, y_pred),
    }


def evaluate_models(
    rows: list[dict],
    feature_cols: list[str],
    target_col: str,
    random_state: int,
) -> tuple[list[dict], list[dict]]:
    x, y = rows_to_arrays(rows, feature_cols, target_col)
    splits, cv_name = make_splits(len(rows), random_state)

    results: list[dict] = []
    prediction_rows: list[dict] = []

    model_names = ("baseline_mean", "linear", "ridge")
    preds_by_model = {name: np.zeros_like(y, dtype=float) for name in model_names}

    for train_idx, test_idx in splits:
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]

        preds_by_model["baseline_mean"][test_idx] = float(np.mean(y_train))

        linear_model = fit_linear_model(x_train, y_train)
        preds_by_model["linear"][test_idx] = predict_model(linear_model, x_test)

        ridge_model = fit_ridge_model(x_train, y_train, random_state=random_state + int(test_idx[0]))
        preds_by_model["ridge"][test_idx] = predict_model(ridge_model, x_test)

    for model_name in model_names:
        metrics = score_predictions(y, preds_by_model[model_name])
        results.append({"model": model_name, "cv": cv_name, "n": len(rows), **metrics})
        for idx, row in enumerate(rows):
            prediction_rows.append(
                {
                    "pid": row["pid"],
                    "side": row["side"],
                    "model": model_name,
                    "y_true": float(y[idx]),
                    "y_pred": float(preds_by_model[model_name][idx]),
                    "residual": float(y[idx] - preds_by_model[model_name][idx]),
                }
            )

    results.sort(key=lambda item: (item["rmse"], item["mae"], -item["r2"] if not math.isnan(item["r2"]) else float("inf")))
    return results, prediction_rows


def fit_final_model(model_name: str, rows: list[dict], feature_cols: list[str], target_col: str, random_state: int) -> dict:
    x, y = rows_to_arrays(rows, feature_cols, target_col)
    if model_name == "linear":
        return fit_linear_model(x, y)
    if model_name == "ridge":
        return fit_ridge_model(x, y, random_state)
    raise ValueError(f"Unsupported final model: {model_name}")


def extract_feature_importance(model_name: str, model: dict, feature_cols: list[str]) -> list[dict]:
    if model_name not in {"linear", "ridge"}:
        return []
    coef = np.asarray(model["coef"], dtype=float)[1:]
    rows = []
    for feature, value in zip(feature_cols, coef):
        rows.append(
            {
                "feature": feature,
                "metric": "coefficient",
                "value": float(value),
                "abs_value": float(abs(value)),
            }
        )
    rows.sort(key=lambda item: item["abs_value"], reverse=True)
    return rows


def save_model_bundle(
    save_path: Path,
    model_name: str,
    model: dict,
    feature_cols: list[str],
    target_name: str,
    metadata: dict,
) -> None:
    bundle = {
        "model_name": model_name,
        "model": model,
        "feature_cols": feature_cols,
        "target_name": target_name,
        "metadata": metadata,
    }
    with save_path.open("wb") as f:
        pickle.dump(bundle, f)


def run_training(args: argparse.Namespace) -> None:
    analysis_dir = infer_best_analysis_dir(args.analysis_dir)
    output_dir = prepare_output_dir(args.output_dir, analysis_dir)

    rows = build_training_rows(analysis_dir, args.target)
    metrics_rows, prediction_rows = evaluate_models(rows, FEATURE_COLS, "target_value", args.random_state)
    best_model_name = metrics_rows[0]["model"]
    if best_model_name == "baseline_mean":
        best_model_name = "ridge" if len(metrics_rows) > 1 else "linear"

    final_model = fit_final_model(best_model_name, rows, FEATURE_COLS, "target_value", args.random_state)
    importance_rows = extract_feature_importance(best_model_name, final_model, FEATURE_COLS)

    feature_csv = output_dir / "angle_ratio_training_dataset.csv"
    metrics_csv = output_dir / "model_comparison.csv"
    pred_csv = output_dir / "cv_predictions.csv"
    importance_csv = output_dir / "feature_importance.csv"
    summary_json = output_dir / "summary.json"
    model_pkl = Path(args.model_path) if args.model_path else output_dir / "best_model.pkl"

    write_csv_rows(
        feature_csv,
        ["pid", "side", *FEATURE_COLS, "target_value"],
        rows,
    )
    write_csv_rows(
        metrics_csv,
        ["model", "cv", "n", "mae", "rmse", "r2", "pearson_r"],
        metrics_rows,
    )
    write_csv_rows(
        pred_csv,
        ["pid", "side", "model", "y_true", "y_pred", "residual"],
        prediction_rows,
    )
    write_csv_rows(
        importance_csv,
        ["feature", "metric", "value", "abs_value"],
        importance_rows,
    )

    best_metrics = next(row for row in metrics_rows if row["model"] == best_model_name)
    summary = {
        "analysis_dir": str(analysis_dir),
        "output_dir": str(output_dir),
        "target": args.target,
        "n_ears": len(rows),
        "feature_cols": FEATURE_COLS,
        "best_model": best_model_name,
        "best_metrics": best_metrics,
        "files": {
            "training_dataset": str(feature_csv),
            "model_comparison": str(metrics_csv),
            "cv_predictions": str(pred_csv),
            "feature_importance": str(importance_csv),
            "model_bundle": str(model_pkl),
        },
        "note": "This script uses pure NumPy so it can run in environments where pandas/sklearn are unavailable.",
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    save_model_bundle(model_pkl, best_model_name, final_model, FEATURE_COLS, args.target, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def load_model_bundle(model_path: Path) -> dict:
    with model_path.open("rb") as f:
        return pickle.load(f)


def run_inference(args: argparse.Namespace) -> None:
    if not args.model_path:
        raise ValueError("--model-path is required when using --predict-angle-csv")

    bundle = load_model_bundle(Path(args.model_path))
    model = bundle["model"]
    feature_cols = list(bundle["feature_cols"])
    target_name = str(bundle["target_name"])

    feature_map = load_angle_features(Path(args.predict_angle_csv))
    rows = []
    for key in sorted(feature_map):
        pid, side = key
        row = {"pid": pid, "side": side, **feature_map[key]}
        rows.append(row)

    x = np.array([[float(row[col]) for col in feature_cols] for row in rows], dtype=float)
    preds = predict_model(model, x)

    out_rows = []
    for row, pred in zip(rows, preds):
        out_rows.append({**row, f"pred_{target_name}": float(pred)})

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.predict_angle_csv).resolve().parent / "angle_ratio_predictions"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "predicted_ratio.csv"
    write_csv_rows(out_csv, ["pid", "side", *feature_cols, f"pred_{target_name}"], out_rows)

    summary = {
        "model_path": str(Path(args.model_path).resolve()),
        "prediction_input": str(Path(args.predict_angle_csv).resolve()),
        "output_csv": str(out_csv.resolve()),
        "n_predictions": len(out_rows),
        "target": target_name,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if args.predict_angle_csv:
        run_inference(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
