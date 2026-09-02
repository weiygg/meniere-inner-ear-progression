from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a PHI-safe aggregate summary of predicted canal features."
    )
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_values(rows: list[dict[str, str]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(field, "")
        if raw in (None, ""):
            continue
        value = float(raw)
        if math.isfinite(value):
            values.append(value)
    return values


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def describe(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    if not ordered:
        return {"n": 0, "mean": None, "sd": None, "median": None, "q1": None, "q3": None}
    return {
        "n": len(ordered),
        "mean": statistics.fmean(ordered),
        "sd": statistics.stdev(ordered) if len(ordered) > 1 else None,
        "median": statistics.median(ordered),
        "q1": percentile(ordered, 0.25),
        "q3": percentile(ordered, 0.75),
    }


def grouped_descriptions(
    rows: list[dict[str, str]],
    group_fields: tuple[str, ...],
    metric_fields: tuple[str, ...],
) -> list[dict]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, group_rows in sorted(grouped.items()):
        item = {field: value for field, value in zip(group_fields, key)}
        item["row_n"] = len(group_rows)
        item["metrics"] = {
            field: describe(finite_values(group_rows, field)) for field in metric_fields
        }
        output.append(item)
    return output


def add_pooled(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    pooled = [dict(row, cohort="Pooled") for row in rows]
    return rows + pooled


def main() -> None:
    args = parse_args()
    feature_dir = args.feature_dir
    features = read_rows(feature_dir / "external_features.csv")
    centerlines = read_rows(feature_dir / "external_centerlines.csv")
    angles = read_rows(feature_dir / "external_plane_angles.csv")
    asymmetry = read_rows(feature_dir / "external_bilateral_asymmetry.csv")
    errors = read_rows(feature_dir / "feature_errors.csv")

    centerline_pass = [row for row in centerlines if row["centerline_status"] == "pass"]
    payload = {
        "schema_version": "1.0",
        "model_label": args.model_label,
        "scope": (
            "Descriptive geometry derived from frozen model-predicted masks in two previously "
            "exposed same-institution external strata; not confirmatory external validation."
        ),
        "privacy": "Aggregate only; no study identifiers, paths, images, masks, or weights.",
        "counts": {
            "patients": len({(row["cohort"], row["study_id"]) for row in features}),
            "ears": len({(row["cohort"], row["study_id"], row["ear_side"]) for row in features}),
            "masks": len(features),
            "centerline_pass": len(centerline_pass),
            "centerline_failed": len(centerlines) - len(centerline_pass),
            "plane_angles": len(angles),
            "bilateral_asymmetry_rows": len(asymmetry),
            "feature_errors": len(errors),
        },
        "shape_by_center_and_structure": grouped_descriptions(
            add_pooled(features),
            ("cohort", "structure"),
            (
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
            ),
        ),
        "centerline_by_center_and_structure": grouped_descriptions(
            add_pooled(centerline_pass),
            ("cohort", "structure"),
            (
                "centerline_length_mm",
                "mean_diameter_mm",
                "minimum_diameter_mm",
                "mean_curvature_per_mm",
                "maximum_curvature_per_mm",
                "mean_abs_torsion_per_mm",
                "plane_rms_residual_mm",
            ),
        ),
        "plane_angle_by_center_and_pair": grouped_descriptions(
            add_pooled(angles),
            ("cohort", "canal_a", "canal_b"),
            ("acute_plane_angle_degrees",),
        ),
        "bilateral_asymmetry_by_center_structure_metric": grouped_descriptions(
            add_pooled(asymmetry),
            ("cohort", "structure", "metric"),
            ("relative_absolute_difference",),
        ),
        "qc_boundary": (
            "Features are outputs of predicted masks. Exclude failed centerlines for centerline and "
            "plane analyses, visually review segmentation QC, and do not use these exposed strata "
            "to tune or select the segmentation model."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
