from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_MORPH_PATH = Path(__file__).resolve().with_name("05_extract_inner_ear_morphometry.py")
_SPEC = importlib.util.spec_from_file_location("inner_ear_morphometry_impl", _MORPH_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Cannot load morphometry implementation: {_MORPH_PATH}")
_MORPH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MORPH)
basic_features = _MORPH.basic_features
centerline_features = _MORPH.centerline_features


ALL_STRUCTURES = ("Cochlear", "Vestibular", "SSC", "HSC", "PSC", "TV")
CANALS = ("SSC", "HSC", "PSC")
BASIC_METRICS = (
    "volume_mm3",
    "surface_area_mm2",
    "surface_to_volume_ratio_per_mm",
    "sphericity",
    "compactness",
    "maximum_3d_diameter_mm",
    "principal_axis_1_mm",
    "principal_axis_2_mm",
    "principal_axis_3_mm",
    "elongation",
    "flatness",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract all-six-structure morphometry and spatial features by center.")
    parser.add_argument(
        "--external-root",
        type=Path,
        default=Path("results_md_progression/final/all_t2_inner_ear_vit_20260801"),
    )
    parser.add_argument("--centers", default="center2,center3")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return float("nan")
    cosine = abs(float(np.dot(a, b))) / denominator
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def cropped_mask_path(source_path: Path, temporary_root: Path, index: int) -> Path:
    image = nib.load(str(source_path))
    binary = np.asarray(image.dataobj) > 0
    coordinates = np.argwhere(binary)
    if coordinates.size == 0:
        raise ValueError("empty mask")
    lower = np.maximum(coordinates.min(axis=0) - 3, 0)
    upper = np.minimum(coordinates.max(axis=0) + 4, np.asarray(binary.shape))
    slices = tuple(slice(int(lower[axis]), int(upper[axis])) for axis in range(3))
    translation = nib.affines.from_matvec(np.eye(3), lower.astype(float))
    cropped_affine = image.affine @ translation
    output_path = temporary_root / f"{index:05d}.nii.gz"
    nib.save(nib.Nifti1Image(binary[slices].astype(np.uint8), cropped_affine), str(output_path))
    return output_path


def flatten_patient_matrix(
    center: str,
    features: list[dict[str, object]],
    centerlines: list[dict[str, object]],
    distances: list[dict[str, object]],
    angles: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for feature in features:
        study_id = str(feature["study_id"])
        row = rows.setdefault(
            study_id,
            {
                "center": center,
                "study_id": study_id,
                "manual_mask_review_status": "pending",
                "analysis_eligible": False,
            },
        )
        side = str(feature["ear_side"])
        structure = str(feature["structure"])
        for metric in BASIC_METRICS:
            row[f"{side}_{structure}_{metric}"] = feature.get(metric)
        row[f"{side}_{structure}_auto_qc_status"] = feature.get("auto_qc_status")
    for line in centerlines:
        if line.get("centerline_status") != "pass":
            continue
        row = rows[str(line["study_id"])]
        prefix = f"{line['ear_side']}_{line['structure']}"
        for metric in (
            "centerline_length_mm",
            "mean_diameter_mm",
            "minimum_diameter_mm",
            "mean_curvature_per_mm",
            "maximum_curvature_per_mm",
            "mean_abs_torsion_per_mm",
            "plane_rms_residual_mm",
        ):
            row[f"{prefix}_{metric}"] = line.get(metric)
    for item in distances:
        row = rows[str(item["study_id"])]
        row[f"{item['ear_side']}_{item['structure_a']}_{item['structure_b']}_centroid_distance_mm"] = item[
            "centroid_distance_mm"
        ]
    for item in angles:
        row = rows[str(item["study_id"])]
        row[f"{item['ear_side']}_{item['canal_a']}_{item['canal_b']}_plane_angle_deg"] = item[
            "acute_plane_angle_degrees"
        ]

    for row in rows.values():
        study_id = str(row["study_id"])
        per_study_features = [feature for feature in features if str(feature["study_id"]) == study_id]
        row["automatic_mask_warning_count"] = sum(feature["auto_qc_status"] != "pass" for feature in per_study_features)
        for structure in ALL_STRUCTURES:
            for metric in ("volume_mm3", "surface_area_mm2", "maximum_3d_diameter_mm"):
                left = row.get(f"L_{structure}_{metric}")
                right = row.get(f"R_{structure}_{metric}")
                if left is None or right is None:
                    continue
                mean = (abs(float(left)) + abs(float(right))) / 2.0
                row[f"{structure}_{metric}_bilateral_relative_absolute_difference"] = (
                    abs(float(left) - float(right)) / mean if mean else None
                )
    return [rows[study_id] for study_id in sorted(rows)]


def main() -> None:
    args = parse_args()
    requested_centers = [item.strip() for item in args.centers.split(",") if item.strip()]
    overall: dict[str, object] = {"centers": {}}
    for center in requested_centers:
        center_dir = args.external_root / f"external_{center}"
        manifest = read_csv(center_dir / "all_six_structure_mask_manifest.csv")
        output_dir = center_dir / "features_pre_manual_qc"
        output_dir.mkdir(parents=True, exist_ok=True)
        features: list[dict[str, object]] = []
        centerlines: list[dict[str, object]] = []
        errors: list[dict[str, object]] = []
        normals: dict[tuple[str, str], dict[str, np.ndarray]] = defaultdict(dict)
        centroids: dict[tuple[str, str], dict[str, np.ndarray]] = defaultdict(dict)
        temporary_context = tempfile.TemporaryDirectory(prefix=f"{center}_feature_crops_")
        temporary_root = Path(temporary_context.name)
        for index, row in enumerate(manifest, start=1):
            path = Path(row["mask_path"])
            study_id = row["study_id"]
            side = row["ear_side"]
            structure = row["structure"]
            try:
                calculation_path = cropped_mask_path(path, temporary_root, index)
                feat, binary, spacing, affine = basic_features(calculation_path)
                axes = feat["principal_axis_lengths_mm"]
                centroid = np.asarray(feat["centroid_mm"], dtype=float)
                feature_row: dict[str, object] = {
                    "center": center,
                    "study_id": study_id,
                    "ear_side": side,
                    "structure": structure,
                    "auto_qc_status": row["qc_status"],
                    "auto_qc_flags": row["qc_flags"],
                    "manual_mask_review_status": "pending",
                    "analysis_eligible": False,
                    "volume_mm3": feat["volume_mm3"],
                    "surface_area_mm2": feat["surface_area_mm2"],
                    "surface_to_volume_ratio_per_mm": feat["surface_to_volume_ratio"],
                    "sphericity": feat["sphericity"],
                    "compactness": feat["compactness"],
                    "maximum_3d_diameter_mm": feat["maximum_3d_diameter_mm"],
                    "principal_axis_1_mm": axes[0],
                    "principal_axis_2_mm": axes[1],
                    "principal_axis_3_mm": axes[2],
                    "elongation": feat["elongation"],
                    "flatness": feat["flatness"],
                    "centroid_x_mm": centroid[0],
                    "centroid_y_mm": centroid[1],
                    "centroid_z_mm": centroid[2],
                    "mask_path": str(path.resolve()),
                }
                features.append(feature_row)
                centroids[(study_id, side)][structure] = centroid
            except Exception as exc:
                errors.append(
                    {
                        "center": center,
                        "study_id": study_id,
                        "ear_side": side,
                        "structure": structure,
                        "step": "basic_features",
                        "error": f"{type(exc).__name__}: {exc}",
                        "mask_path": str(path.resolve()),
                    }
                )
                continue
            if structure in CANALS:
                try:
                    canal = centerline_features(binary, spacing, affine)
                    normal = np.asarray(canal["plane_normal"], dtype=float)
                    centerlines.append(
                        {
                            "center": center,
                            "study_id": study_id,
                            "ear_side": side,
                            "structure": structure,
                            "centerline_status": "pass",
                            "failure_reason": "",
                            "centerline_length_mm": canal["centerline_length_mm"],
                            "mean_diameter_mm": canal["mean_diameter_mm"],
                            "minimum_diameter_mm": canal["minimum_diameter_mm"],
                            "mean_curvature_per_mm": canal["mean_curvature_per_mm"],
                            "maximum_curvature_per_mm": canal["maximum_curvature_per_mm"],
                            "mean_abs_torsion_per_mm": canal["mean_abs_torsion_per_mm"],
                            "plane_rms_residual_mm": canal["plane_rms_residual_mm"],
                            "plane_normal_x": normal[0],
                            "plane_normal_y": normal[1],
                            "plane_normal_z": normal[2],
                            "skeleton_voxel_n": canal["skeleton_voxel_n"],
                            "main_path_point_n": canal["main_path_point_n"],
                            "closed_loop_skeleton": canal["closed_loop_skeleton"],
                        }
                    )
                    normals[(study_id, side)][structure] = normal
                except Exception as exc:
                    centerlines.append(
                        {
                            "center": center,
                            "study_id": study_id,
                            "ear_side": side,
                            "structure": structure,
                            "centerline_status": "failed",
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    errors.append(
                        {
                            "center": center,
                            "study_id": study_id,
                            "ear_side": side,
                            "structure": structure,
                            "step": "centerline_features",
                            "error": f"{type(exc).__name__}: {exc}",
                            "mask_path": str(path.resolve()),
                        }
                    )
            if index % 100 == 0 or index == len(manifest):
                print(f"FEATURE_PROGRESS {center} {index}/{len(manifest)}", flush=True)
        temporary_context.cleanup()

        distances: list[dict[str, object]] = []
        for (study_id, side), structure_centroids in sorted(centroids.items()):
            for first, second in itertools.combinations(ALL_STRUCTURES, 2):
                if first not in structure_centroids or second not in structure_centroids:
                    continue
                distances.append(
                    {
                        "center": center,
                        "study_id": study_id,
                        "ear_side": side,
                        "structure_a": first,
                        "structure_b": second,
                        "centroid_distance_mm": float(np.linalg.norm(structure_centroids[first] - structure_centroids[second])),
                    }
                )
        angles: list[dict[str, object]] = []
        for (study_id, side), structure_normals in sorted(normals.items()):
            for first, second in itertools.combinations(CANALS, 2):
                if first not in structure_normals or second not in structure_normals:
                    continue
                angles.append(
                    {
                        "center": center,
                        "study_id": study_id,
                        "ear_side": side,
                        "canal_a": first,
                        "canal_b": second,
                        "acute_plane_angle_degrees": angle_degrees(structure_normals[first], structure_normals[second]),
                    }
                )
        matrix = flatten_patient_matrix(center, features, centerlines, distances, angles)
        write_csv(output_dir / "structure_features_long.csv", features)
        write_csv(output_dir / "canal_centerline_features.csv", centerlines)
        write_csv(output_dir / "within_ear_centroid_distances.csv", distances)
        write_csv(output_dir / "canal_plane_angles.csv", angles)
        write_csv(output_dir / "patient_feature_matrix_pending_manual_qc.csv", matrix)
        write_csv(output_dir / "feature_errors.csv", errors)
        summary = {
            "center": center,
            "mask_manifest_rows": len(manifest),
            "basic_feature_rows": len(features),
            "centerline_rows": len(centerlines),
            "centerline_pass": sum(row["centerline_status"] == "pass" for row in centerlines),
            "centroid_distance_rows": len(distances),
            "plane_angle_rows": len(angles),
            "patient_matrix_rows": len(matrix),
            "error_rows": len(errors),
            "analysis_eligible_rows": 0,
            "interpretation_boundary": "Feature extraction is complete, but every external row remains ineligible until manual mask review.",
        }
        (output_dir / "feature_extraction_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        overall["centers"][center] = summary
    (args.external_root / "external_feature_extraction_summary.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("ALL_T2_FEATURE_EXTRACTION_COMPLETE", json.dumps(overall, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
