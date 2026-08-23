from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import ball, binary_closing, skeletonize


STRUCTS_CANALS = ("SSC", "HSC", "PSC")
STRUCTS_VOLUME = ("Cochlear", "Vestibular", "SSC", "HSC", "PSC", "TV", "ELS")
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
FNAME_PATTERN = re.compile(
    r"(?P<pid>\d+)(?P<side>[LR])[-_](?P<struct>[A-Za-z]+)\.nii(\.gz)?$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract robust inner-ear geometry features for modeling.")
    parser.add_argument("--data-dir", required=True, help="Dataset folder containing subject subfolders.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output folder. Default: <data-dir>/analysis_out_geometry",
    )
    parser.add_argument("--min-component-voxels", type=int, default=200)
    parser.add_argument("--closing-radius", type=int, default=1)
    parser.add_argument("--skeleton-min-points", type=int, default=20)
    parser.add_argument(
        "--fallback-min-mask-points",
        type=int,
        default=150,
        help="Fallback to voxel cloud plane fitting when the skeleton is too short.",
    )
    parser.add_argument(
        "--subsample-mask-points",
        type=int,
        default=4000,
        help="Subsample voxel cloud points for faster PCA when needed.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def canonical_struct(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z]", "", name).upper()
    return STRUCT_ALIASES.get(cleaned, cleaned.title())


def load_mask_nii(fp: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = nib.load(str(fp))
    img = nib.as_closest_canonical(img)
    data = img.get_fdata()
    mask = (data > 0.5).astype(np.uint8)
    zooms = tuple(float(v) for v in img.header.get_zooms()[:3])
    return mask, zooms


def keep_largest_component(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    lab = label(mask)
    if lab.max() == 0:
        return mask
    props = sorted(regionprops(lab), key=lambda r: r.area, reverse=True)
    largest = props[0]
    if largest.area < min_voxels:
        return mask
    return (lab == largest.label).astype(np.uint8)


def clean_mask(mask: np.ndarray, closing_radius: int, min_voxels: int) -> np.ndarray:
    if mask.sum() == 0:
        return mask
    closed = binary_closing(mask.astype(bool), ball(closing_radius)).astype(np.uint8)
    return keep_largest_component(closed, min_voxels=min_voxels)


def volume_mm3(mask: np.ndarray, zooms: tuple[float, float, float]) -> float:
    return float(mask.sum() * float(zooms[0] * zooms[1] * zooms[2]))


def mask_points_mm(mask: np.ndarray, zooms: tuple[float, float, float]) -> np.ndarray:
    pts = np.argwhere(mask > 0)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=float)
    return pts.astype(float) * np.asarray(zooms, dtype=float)[None, :]


def skeleton_points_mm(mask: np.ndarray, zooms: tuple[float, float, float]) -> np.ndarray:
    if mask.sum() == 0:
        return np.zeros((0, 3), dtype=float)
    sk = skeletonize(mask.astype(bool), method="lee")
    pts = np.argwhere(sk > 0)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=float)
    return pts.astype(float) * np.asarray(zooms, dtype=float)[None, :]


def subsample_points(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def fit_plane_pca(points_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points_mm.mean(axis=0)
    x = points_mm - center
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    normal = vt[-1, :]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    return center, normal


def fit_pca_shape(points_mm: np.ndarray) -> dict[str, float]:
    center = points_mm.mean(axis=0)
    x = points_mm - center
    cov = np.cov(x, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0.0, None)
    eigvecs = eigvecs[:, order]
    axes_mm = 4.0 * np.sqrt(eigvals + 1e-12)
    l1, l2, l3 = [float(v) for v in eigvals]
    return {
        "pca_eval1": l1,
        "pca_eval2": l2,
        "pca_eval3": l3,
        "axis1_mm": float(axes_mm[0]),
        "axis2_mm": float(axes_mm[1]),
        "axis3_mm": float(axes_mm[2]),
        "linearity": float((l1 - l2) / (l1 + 1e-12)),
        "planarity": float((l2 - l3) / (l1 + 1e-12)),
        "scattering": float(l3 / (l1 + 1e-12)),
        "main_dir_x": float(eigvecs[0, 0]),
        "main_dir_y": float(eigvecs[1, 0]),
        "main_dir_z": float(eigvecs[2, 0]),
        "shape_cx": float(center[0]),
        "shape_cy": float(center[1]),
        "shape_cz": float(center[2]),
    }


def plane_residuals(points_mm: np.ndarray, center: np.ndarray, normal: np.ndarray) -> tuple[float, float]:
    distances = np.abs((points_mm - center[None, :]) @ normal)
    rmse = float(np.sqrt(np.mean(distances**2)))
    mae = float(np.mean(distances))
    return rmse, mae


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    anchor = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(anchor, normal))) > 0.9:
        anchor = np.array([0.0, 1.0, 0.0], dtype=float)
    u = np.cross(normal, anchor)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(normal, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    return u, v


def circular_order(projected_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = projected_xy - projected_xy.mean(axis=0, keepdims=True)
    raw_angles = np.arctan2(centered[:, 1], centered[:, 0])
    sort_idx = np.argsort(raw_angles)
    sorted_angles = raw_angles[sort_idx]
    wrapped = np.concatenate([sorted_angles, sorted_angles[:1] + 2.0 * np.pi])
    gaps = np.diff(wrapped)
    cut = int(np.argmax(gaps))
    ordered_idx = np.concatenate([sort_idx[cut + 1 :], sort_idx[: cut + 1]])
    ordered_angles = raw_angles[ordered_idx]
    for idx in range(1, ordered_angles.shape[0]):
        while ordered_angles[idx] < ordered_angles[idx - 1]:
            ordered_angles[idx] += 2.0 * np.pi
    return ordered_idx, ordered_angles


def projected_arc_features(points_mm: np.ndarray, center: np.ndarray, normal: np.ndarray) -> dict[str, float]:
    if points_mm.shape[0] < 3:
        return {
            "proj_radius_mean_mm": float("nan"),
            "proj_radius_std_mm": float("nan"),
            "proj_arc_span_deg": float("nan"),
            "proj_arc_length_mm": float("nan"),
            "proj_chord_length_mm": float("nan"),
            "proj_arc_chord_ratio": float("nan"),
        }
    u, v = plane_basis(normal)
    rel = points_mm - center[None, :]
    projected = np.stack([rel @ u, rel @ v], axis=1)
    centered = projected - projected.mean(axis=0, keepdims=True)
    radii = np.linalg.norm(centered, axis=1)
    ordered_idx, ordered_angles = circular_order(projected)
    ordered = projected[ordered_idx]
    if ordered.shape[0] >= 2:
        arc_length = float(np.linalg.norm(np.diff(ordered, axis=0), axis=1).sum())
        chord_length = float(np.linalg.norm(ordered[-1] - ordered[0]))
    else:
        arc_length = float("nan")
        chord_length = float("nan")
    ratio = float(arc_length / chord_length) if chord_length and chord_length > 1e-8 else float("nan")
    return {
        "proj_radius_mean_mm": float(np.mean(radii)),
        "proj_radius_std_mm": float(np.std(radii)),
        "proj_arc_span_deg": float(np.degrees(ordered_angles[-1] - ordered_angles[0])),
        "proj_arc_length_mm": arc_length,
        "proj_chord_length_mm": chord_length,
        "proj_arc_chord_ratio": ratio,
    }


def angle_between_normals(n1: np.ndarray, n2: np.ndarray) -> float:
    n1 = n1 / (np.linalg.norm(n1) + 1e-12)
    n2 = n2 / (np.linalg.norm(n2) + 1e-12)
    cos_value = np.clip(np.abs(np.dot(n1, n2)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_value)))


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_files(data_dir: Path) -> list[dict]:
    items = []
    for fp in sorted(data_dir.rglob("*.nii*")):
        match = FNAME_PATTERN.match(fp.name)
        if not match:
            continue
        filename_pid = match.group("pid")
        parent_match = re.match(r"^sub(?P<pid>\d+)$", fp.parent.name, re.IGNORECASE)
        pid = parent_match.group("pid") if parent_match else filename_pid
        items.append(
            {
                "pid": pid,
                "filename_pid": filename_pid,
                "pid_from_parent_dir": int(parent_match is not None and parent_match.group("pid") != filename_pid),
                "side": match.group("side").upper(),
                "struct": canonical_struct(match.group("struct")),
                "path": fp,
            }
        )
    return items


def build_ear_feature_row(pid: str, side: str, canal_rows: dict[str, dict]) -> dict:
    row: dict[str, object] = {
        "pid": pid,
        "side": side,
        "complete_three_canals": int(all(struct in canal_rows for struct in STRUCTS_CANALS)),
    }
    for struct in STRUCTS_CANALS:
        prefix = struct.lower()
        canal = canal_rows.get(struct)
        if not canal:
            continue
        row[f"{prefix}_volume_mm3"] = canal["volume_mm3"]
        row[f"{prefix}_mask_points"] = canal["mask_points"]
        row[f"{prefix}_skeleton_points"] = canal["skeleton_points"]
        row[f"{prefix}_plane_source"] = canal["plane_source"]
        row[f"{prefix}_plane_source_is_skeleton"] = int(canal["plane_source"] == "skeleton")
        row[f"{prefix}_nx"] = canal["nx"]
        row[f"{prefix}_ny"] = canal["ny"]
        row[f"{prefix}_nz"] = canal["nz"]
        row[f"{prefix}_abs_nx"] = abs(canal["nx"])
        row[f"{prefix}_abs_ny"] = abs(canal["ny"])
        row[f"{prefix}_abs_nz"] = abs(canal["nz"])
        row[f"{prefix}_cx"] = canal["cx"]
        row[f"{prefix}_cy"] = canal["cy"]
        row[f"{prefix}_cz"] = canal["cz"]
        row[f"{prefix}_plane_rmse_mm"] = canal["plane_rmse_mm"]
        row[f"{prefix}_plane_mae_mm"] = canal["plane_mae_mm"]
        row[f"{prefix}_axis1_mm"] = canal["axis1_mm"]
        row[f"{prefix}_axis2_mm"] = canal["axis2_mm"]
        row[f"{prefix}_axis3_mm"] = canal["axis3_mm"]
        row[f"{prefix}_linearity"] = canal["linearity"]
        row[f"{prefix}_planarity"] = canal["planarity"]
        row[f"{prefix}_scattering"] = canal["scattering"]
        row[f"{prefix}_proj_radius_mean_mm"] = canal["proj_radius_mean_mm"]
        row[f"{prefix}_proj_radius_std_mm"] = canal["proj_radius_std_mm"]
        row[f"{prefix}_proj_arc_span_deg"] = canal["proj_arc_span_deg"]
        row[f"{prefix}_proj_arc_length_mm"] = canal["proj_arc_length_mm"]
        row[f"{prefix}_proj_chord_length_mm"] = canal["proj_chord_length_mm"]
        row[f"{prefix}_proj_arc_chord_ratio"] = canal["proj_arc_chord_ratio"]

    if all(struct in canal_rows for struct in STRUCTS_CANALS):
        normals = {struct: np.array([canal_rows[struct]["nx"], canal_rows[struct]["ny"], canal_rows[struct]["nz"]]) for struct in STRUCTS_CANALS}
        centers = {struct: np.array([canal_rows[struct]["cx"], canal_rows[struct]["cy"], canal_rows[struct]["cz"]]) for struct in STRUCTS_CANALS}
        row["angle_ssc_hsc_deg"] = angle_between_normals(normals["SSC"], normals["HSC"])
        row["angle_ssc_psc_deg"] = angle_between_normals(normals["SSC"], normals["PSC"])
        row["angle_hsc_psc_deg"] = angle_between_normals(normals["HSC"], normals["PSC"])
        row["orthogonality_deviation_mean_deg"] = float(
            np.mean(
                [
                    abs(row["angle_ssc_hsc_deg"] - 90.0),
                    abs(row["angle_ssc_psc_deg"] - 90.0),
                    abs(row["angle_hsc_psc_deg"] - 90.0),
                ]
            )
        )
        row["center_dist_ssc_hsc_mm"] = float(np.linalg.norm(centers["SSC"] - centers["HSC"]))
        row["center_dist_ssc_psc_mm"] = float(np.linalg.norm(centers["SSC"] - centers["PSC"]))
        row["center_dist_hsc_psc_mm"] = float(np.linalg.norm(centers["HSC"] - centers["PSC"]))
    return row


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else data_dir / "analysis_out_geometry"
    output_dir.mkdir(parents=True, exist_ok=True)

    file_rows = parse_files(data_dir)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in file_rows:
        grouped[(row["pid"], row["side"])].append(row)

    volume_rows: list[dict] = []
    canal_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    ear_rows: list[dict] = []

    for (pid, side), sub_rows in sorted(grouped.items(), key=lambda x: (int(x[0][0]), x[0][1])):
        rows_by_struct: dict[str, list[dict]] = defaultdict(list)
        for row in sub_rows:
            rows_by_struct[row["struct"]].append(row)

        canal_feature_map: dict[str, dict] = {}

        for struct in STRUCTS_VOLUME:
            recs = rows_by_struct.get(struct, [])
            if not recs:
                continue
            fp = recs[0]["path"]
            mask, zooms = load_mask_nii(fp)
            cleaned = clean_mask(mask, closing_radius=args.closing_radius, min_voxels=args.min_component_voxels)
            volume_rows.append(
                {
                    "pid": pid,
                    "side": side,
                    "struct": struct,
                    "volume_mm3": round(volume_mm3(cleaned, zooms), 6),
                    "file": str(fp),
                }
            )

        for struct in STRUCTS_CANALS:
            recs = rows_by_struct.get(struct, [])
            if not recs:
                diagnostic_rows.append({"pid": pid, "side": side, "struct": struct, "status": "missing_file"})
                continue

            fp = recs[0]["path"]
            mask, zooms = load_mask_nii(fp)
            cleaned = clean_mask(mask, closing_radius=args.closing_radius, min_voxels=args.min_component_voxels)
            vox_points = mask_points_mm(cleaned, zooms)
            skel_points = skeleton_points_mm(cleaned, zooms)
            chosen_points = None
            source = "missing"
            if skel_points.shape[0] >= args.skeleton_min_points:
                chosen_points = skel_points
                source = "skeleton"
            elif vox_points.shape[0] >= args.fallback_min_mask_points:
                chosen_points = subsample_points(vox_points, args.subsample_mask_points, rng)
                source = "mask"

            diagnostic_row = {
                "pid": pid,
                "side": side,
                "struct": struct,
                "status": "ok" if chosen_points is not None else "insufficient_points",
                "mask_points": int(vox_points.shape[0]),
                "skeleton_points": int(skel_points.shape[0]),
                "plane_source": source,
                "file": str(fp),
            }

            if chosen_points is None or chosen_points.shape[0] < 3:
                diagnostic_rows.append(diagnostic_row)
                continue

            center, normal = fit_plane_pca(chosen_points)
            shape_points = skel_points if skel_points.shape[0] >= 5 else chosen_points
            shape_features = fit_pca_shape(shape_points)
            plane_rmse, plane_mae = plane_residuals(chosen_points, center, normal)
            arc_features = projected_arc_features(shape_points, center, normal)

            canal_row = {
                "pid": pid,
                "side": side,
                "struct": struct,
                "plane_source": source,
                "mask_points": int(vox_points.shape[0]),
                "skeleton_points": int(skel_points.shape[0]),
                "volume_mm3": round(volume_mm3(cleaned, zooms), 6),
                "nx": float(normal[0]),
                "ny": float(normal[1]),
                "nz": float(normal[2]),
                "cx": float(center[0]),
                "cy": float(center[1]),
                "cz": float(center[2]),
                "plane_rmse_mm": plane_rmse,
                "plane_mae_mm": plane_mae,
                **shape_features,
                **arc_features,
                "file": str(fp),
            }
            canal_rows.append(canal_row)
            diagnostic_rows.append(diagnostic_row)
            canal_feature_map[struct] = canal_row

        ear_rows.append(build_ear_feature_row(pid, side, canal_feature_map))

    angle_rows: list[dict] = []
    for row in ear_rows:
        if int(row.get("complete_three_canals", 0)) != 1:
            continue
        angle_rows.extend(
            [
                {"pid": row["pid"], "side": row["side"], "angle_type": "SSC-HSC", "angle_deg": row["angle_ssc_hsc_deg"]},
                {"pid": row["pid"], "side": row["side"], "angle_type": "SSC-PSC", "angle_deg": row["angle_ssc_psc_deg"]},
                {"pid": row["pid"], "side": row["side"], "angle_type": "HSC-PSC", "angle_deg": row["angle_hsc_psc_deg"]},
            ]
        )

    write_csv(volume_rows, output_dir / "volumes_mm3.csv")
    write_csv(canal_rows, output_dir / "canal_geometry_features.csv")
    write_csv(ear_rows, output_dir / "ear_geometry_features.csv")
    write_csv(diagnostic_rows, output_dir / "extraction_diagnostics.csv")
    write_csv(angle_rows, output_dir / "canal_plane_angles_deg.csv")
    if canal_rows:
        write_csv(
            [
                {
                    "pid": row["pid"],
                    "side": row["side"],
                    "struct": row["struct"],
                    "nx": row["nx"],
                    "ny": row["ny"],
                    "nz": row["nz"],
                    "cx": row["cx"],
                    "cy": row["cy"],
                    "cz": row["cz"],
                    "n_points": row["skeleton_points"] if row["plane_source"] == "skeleton" else row["mask_points"],
                    "plane_source": row["plane_source"],
                    "file": row["file"],
                }
                for row in canal_rows
            ],
            output_dir / "plane_normals.csv",
        )

    complete_ears = sum(int(row.get("complete_three_canals", 0)) for row in ear_rows)
    summary = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "matched_files": len(file_rows),
        "canal_rows": len(canal_rows),
        "ear_rows": len(ear_rows),
        "complete_three_canal_ears": complete_ears,
        "skeleton_min_points": args.skeleton_min_points,
        "fallback_min_mask_points": args.fallback_min_mask_points,
        "plane_source_counts": {
            "skeleton": sum(1 for row in canal_rows if row["plane_source"] == "skeleton"),
            "mask": sum(1 for row in canal_rows if row["plane_source"] == "mask"),
        },
        "files": {
            "volumes_mm3": str(output_dir / "volumes_mm3.csv"),
            "canal_geometry_features": str(output_dir / "canal_geometry_features.csv"),
            "ear_geometry_features": str(output_dir / "ear_geometry_features.csv"),
            "extraction_diagnostics": str(output_dir / "extraction_diagnostics.csv"),
            "plane_normals": str(output_dir / "plane_normals.csv"),
            "angles": str(output_dir / "canal_plane_angles_deg.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
