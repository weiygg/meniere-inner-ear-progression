from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import re
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


MASK_RE = re.compile(r"^(?P<study>.+)(?P<side>[LR])_(?P<structure>SSC|HSC|PSC)\.nii\.gz$", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract morphometry from frozen external predicted canal masks.")
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    cosine = abs(float(np.dot(a, b))) / float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def cropped_mask_path(source_path: Path, temporary_root: Path, index: int) -> Path:
    image = nib.load(str(source_path))
    binary = np.asarray(image.dataobj) > 0
    coords = np.argwhere(binary)
    if coords.size == 0:
        raise ValueError("empty mask")
    lower = np.maximum(coords.min(axis=0) - 3, 0)
    upper = np.minimum(coords.max(axis=0) + 4, np.asarray(binary.shape))
    slices = tuple(slice(int(lower[axis]), int(upper[axis])) for axis in range(3))
    cropped = binary[slices].astype(np.uint8)
    translation = nib.affines.from_matvec(np.eye(3), lower.astype(float))
    cropped_affine = image.affine @ translation
    temporary_path = temporary_root / f"{index:04d}.nii.gz"
    nib.save(nib.Nifti1Image(cropped, cropped_affine), str(temporary_path))
    return temporary_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = []
    centerlines = []
    errors = []
    normals: dict[tuple[str, str], dict[str, np.ndarray]] = defaultdict(dict)
    by_study_structure: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    files = sorted(args.mask_root.glob("sub*/*.nii.gz"))
    temporary_context = tempfile.TemporaryDirectory(prefix="canal_feature_crops_")
    temporary_root = Path(temporary_context.name)
    for index, path in enumerate(files, start=1):
        match = MASK_RE.match(path.name)
        if not match:
            continue
        study_id = match.group("study")
        side = match.group("side").upper()
        structure = match.group("structure").upper()
        try:
            calculation_path = cropped_mask_path(path, temporary_root, index)
            feat, binary, spacing, affine = basic_features(calculation_path)
            axes = feat["principal_axis_lengths_mm"]
            centroid = feat["centroid_mm"]
            feature_row = {
                "cohort": "Z2_external_predicted",
                "study_id": study_id,
                "ear_side": side,
                "structure": structure,
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
            by_study_structure[(study_id, structure)][side] = feature_row
        except Exception as exc:
            errors.append(
                {
                    "study_id": study_id,
                    "ear_side": side,
                    "structure": structure,
                    "step": "basic_features",
                    "error": f"{type(exc).__name__}: {exc}",
                    "mask_path": str(path.resolve()),
                }
            )
            continue
        try:
            canal = centerline_features(binary, spacing, affine)
            normal = canal["plane_normal"]
            centerlines.append(
                {
                    "cohort": "Z2_external_predicted",
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
            normals[(study_id, side)][structure] = np.asarray(normal)
        except Exception as exc:
            centerlines.append(
                {
                    "cohort": "Z2_external_predicted",
                    "study_id": study_id,
                    "ear_side": side,
                    "structure": structure,
                    "centerline_status": "failed",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )
            errors.append(
                {
                    "study_id": study_id,
                    "ear_side": side,
                    "structure": structure,
                    "step": "centerline_features",
                    "error": f"{type(exc).__name__}: {exc}",
                    "mask_path": str(path.resolve()),
                }
            )
        if index % 25 == 0 or index == len(files):
            print(f"FEATURE_PROGRESS {index}/{len(files)}", flush=True)
    temporary_context.cleanup()

    angles = []
    for (study_id, side), structure_normals in sorted(normals.items()):
        for a, b in itertools.combinations(sorted(structure_normals), 2):
            angles.append(
                {
                    "cohort": "Z2_external_predicted",
                    "study_id": study_id,
                    "ear_side": side,
                    "canal_a": a,
                    "canal_b": b,
                    "acute_plane_angle_degrees": angle_degrees(structure_normals[a], structure_normals[b]),
                }
            )

    asymmetry = []
    for (study_id, structure), sides in sorted(by_study_structure.items()):
        if not {"L", "R"}.issubset(sides):
            continue
        for metric in ("volume_mm3", "surface_area_mm2", "maximum_3d_diameter_mm"):
            left = float(sides["L"][metric])
            right = float(sides["R"][metric])
            mean = (abs(left) + abs(right)) / 2.0
            asymmetry.append(
                {
                    "cohort": "Z2_external_predicted",
                    "study_id": study_id,
                    "structure": structure,
                    "metric": metric,
                    "left_value": left,
                    "right_value": right,
                    "relative_absolute_difference": abs(left - right) / mean if mean else None,
                }
            )

    outputs = {
        "external_features": features,
        "external_centerlines": centerlines,
        "external_plane_angles": angles,
        "external_bilateral_asymmetry": asymmetry,
        "feature_errors": errors,
    }
    for name, rows in outputs.items():
        write_csv(args.output_dir / f"{name}.csv", rows)
    (args.output_dir / "external_features.json").write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "mask_files": len(files),
        "basic_feature_rows": len(features),
        "centerline_rows": len(centerlines),
        "centerline_pass": sum(row["centerline_status"] == "pass" for row in centerlines),
        "plane_angle_rows": len(angles),
        "bilateral_asymmetry_rows": len(asymmetry),
        "error_rows": len(errors),
        "interpretation_boundary": (
            "All Zhejiang Second Hospital features are derived from model-predicted masks and "
            "must be filtered by mask and centerline QC before clinical/statistical use."
        ),
    }
    (args.output_dir / "feature_extraction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("FEATURE_EXTRACTION_COMPLETE", json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
