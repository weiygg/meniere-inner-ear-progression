from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from nibabel.processing import resample_from_to

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meniere_progression.segmentation.geometry import (
    GEOMETRY_ALGORITHM_VERSION,
    inter_canal_angle_degrees,
    mask_geometry,
    plane_normal,
)
from meniere_progression.segmentation.reliability import paired_reliability


FEATURES = ("volume_mm3", "component_count", "centerline_voxel_count", "centerline_length_mm")
ANGLE_PAIRS = (("SSC", "HSC"), ("SSC", "PSC"), ("HSC", "PSC"))


def load_pair(
    reference_path: Path, prediction_path: Path
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], np.ndarray]:
    prediction_image = nib.load(str(prediction_path))
    reference_image = nib.load(str(reference_path))
    prediction_full = np.asarray(prediction_image.dataobj) > 0
    if reference_image.shape == prediction_image.shape and np.allclose(
        reference_image.affine, prediction_image.affine, rtol=1e-5, atol=1e-4
    ):
        prediction = prediction_full
        reference = np.asarray(reference_image.dataobj) > 0
        evaluation_affine = prediction_image.affine
    else:
        reference_source = np.asarray(reference_image.dataobj) > 0
        reference_coordinates = np.argwhere(reference_source)
        prediction_coordinates = np.argwhere(prediction_full)
        if not len(reference_coordinates) or not len(prediction_coordinates):
            reference_image = resample_from_to(
                reference_image, (prediction_image.shape, prediction_image.affine), order=0
            )
            prediction = prediction_full
            reference = np.asarray(reference_image.dataobj) > 0
            evaluation_affine = prediction_image.affine
        else:
            reference_world = nib.affines.apply_affine(reference_image.affine, reference_coordinates)
            reference_target = nib.affines.apply_affine(
                np.linalg.inv(prediction_image.affine), reference_world
            )
            lower = np.floor(
                np.minimum(reference_target.min(axis=0), prediction_coordinates.min(axis=0))
            ).astype(int) - 2
            upper = np.ceil(
                np.maximum(reference_target.max(axis=0), prediction_coordinates.max(axis=0))
            ).astype(int) + 3
            lower = np.maximum(lower, 0)
            upper = np.minimum(upper, np.asarray(prediction_image.shape))
            local_shape = tuple(int(value) for value in upper - lower)
            translation = np.eye(4)
            translation[:3, 3] = lower
            evaluation_affine = prediction_image.affine @ translation
            reference_local = resample_from_to(
                reference_image, (local_shape, evaluation_affine), order=0
            )
            slices = tuple(slice(int(lower[index]), int(upper[index])) for index in range(3))
            prediction = prediction_full[slices]
            reference = np.asarray(reference_local.dataobj) > 0
    spacing = tuple(float(value) for value in prediction_image.header.get_zooms()[:3])
    return reference, prediction, spacing, evaluation_affine


def bootstrap_icc(
    frame: pd.DataFrame, feature: str, repetitions: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    groups = [block for _, block in frame.groupby("patient_key", sort=True)]
    estimates: list[float] = []
    for _ in range(repetitions):
        sampled = pd.concat([groups[index] for index in rng.integers(0, len(groups), len(groups))])
        estimate = paired_reliability(
            sampled[f"manual_{feature}"].to_numpy(), sampled[f"ai_{feature}"].to_numpy()
        ).icc_a1
        if np.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.percentile(estimates, [2.5, 97.5]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate manual-vs-AI canal geometry reliability.")
    parser.add_argument("metrics_csv", type=Path, help="Protected local external mask metric table")
    parser.add_argument("output_dir", type=Path, help="Git-ignored local output directory")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/segmentation_experiments.yaml")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.metrics_csv, dtype={"patient_id": str})
    required = {"cohort", "patient_id", "ear_side", "structure", "manual_mask_path", "prediction_path"}
    if missing := required - set(source):
        raise ValueError(f"Input table missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for index, row in source.iterrows():
        reference, prediction, spacing, affine = load_pair(
            Path(str(row["manual_mask_path"])), Path(str(row["prediction_path"]))
        )
        manual = mask_geometry(reference, spacing)
        automatic = mask_geometry(prediction, spacing)
        record: dict[str, object] = {
            "patient_key": f"{row['cohort']}::{row['patient_id']}",
            "ear_side": row["ear_side"],
            "structure": row["structure"],
        }
        for feature in FEATURES:
            record[f"manual_{feature}"] = manual[feature]
            record[f"ai_{feature}"] = automatic[feature]
        for index, value in enumerate(plane_normal(reference, affine)):
            record[f"manual_plane_normal_{index}"] = value
        for index, value in enumerate(plane_normal(prediction, affine)):
            record[f"ai_plane_normal_{index}"] = value
        rows.append(record)
        if (index + 1) % 50 == 0 or index + 1 == len(source):
            print(f"GEOMETRY_PROGRESS {index + 1}/{len(source)}", flush=True)
    paired = pd.DataFrame(rows)
    paired.to_csv(args.output_dir / "paired_geometry_features_local.csv", index=False, encoding="utf-8-sig")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    gate = config["reliability_gate"]
    angle_rows: list[dict[str, object]] = []
    for (patient_key, ear_side), block in paired.groupby(["patient_key", "ear_side"], sort=True):
        lookup = block.set_index("structure")
        for structure_a, structure_b in ANGLE_PAIRS:
            if structure_a not in lookup.index or structure_b not in lookup.index:
                continue
            manual_a = lookup.loc[structure_a, [f"manual_plane_normal_{i}" for i in range(3)]].to_numpy(float)
            manual_b = lookup.loc[structure_b, [f"manual_plane_normal_{i}" for i in range(3)]].to_numpy(float)
            ai_a = lookup.loc[structure_a, [f"ai_plane_normal_{i}" for i in range(3)]].to_numpy(float)
            ai_b = lookup.loc[structure_b, [f"ai_plane_normal_{i}" for i in range(3)]].to_numpy(float)
            angle_rows.append(
                {
                    "patient_key": patient_key,
                    "ear_side": ear_side,
                    "structure": f"{structure_a}_{structure_b}",
                    "manual_inter_canal_angle_deg": inter_canal_angle_degrees(manual_a, manual_b),
                    "ai_inter_canal_angle_deg": inter_canal_angle_degrees(ai_a, ai_b),
                }
            )
    angles = pd.DataFrame(angle_rows)
    angles.to_csv(args.output_dir / "paired_inter_canal_angles_local.csv", index=False, encoding="utf-8-sig")

    def finite_or_none(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    summary_rows: list[dict[str, object]] = []

    def append_summary(block: pd.DataFrame, structure: str, feature: str, seed_offset: int) -> None:
        block = block.loc[
            np.isfinite(block[f"manual_{feature}"]) & np.isfinite(block[f"ai_{feature}"])
        ]
        estimate = paired_reliability(
            block[f"manual_{feature}"].to_numpy(), block[f"ai_{feature}"].to_numpy()
        )
        ci_low, ci_high = bootstrap_icc(
            block, feature, args.bootstrap, args.seed + seed_offset
        )
        pass_fail = (
            "protocol_blocker_threshold_not_signed"
            if gate["pass_threshold"] is None
            else ("pass" if estimate.icc_a1 >= float(gate["pass_threshold"]) else "fail")
        )
        summary_rows.append(
            {
                "structure": structure,
                "feature": feature,
                "ear_n": len(block),
                "patient_n": block["patient_key"].nunique(),
                "ICC_A1": finite_or_none(estimate.icc_a1),
                "ICC_ci95_low": finite_or_none(ci_low),
                "ICC_ci95_high": finite_or_none(ci_high),
                "bland_altman_bias": estimate.bland_altman_bias,
                "bland_altman_lower": estimate.bland_altman_lower,
                "bland_altman_upper": estimate.bland_altman_upper,
                "mean_absolute_error": estimate.mean_absolute_error,
                "mean_relative_error": estimate.mean_relative_error,
                "reliability_gate": pass_fail,
            }
        )
        mean = (
            block[f"manual_{feature}"].to_numpy() + block[f"ai_{feature}"].to_numpy()
        ) / 2.0
        difference = block[f"ai_{feature}"].to_numpy() - block[f"manual_{feature}"].to_numpy()
        figure, axis = plt.subplots(figsize=(5.5, 4.2))
        axis.scatter(mean, difference, s=18, alpha=0.65)
        axis.axhline(estimate.bland_altman_bias, color="black", linewidth=1.2)
        axis.axhline(estimate.bland_altman_lower, color="tab:red", linestyle="--")
        axis.axhline(estimate.bland_altman_upper, color="tab:red", linestyle="--")
        axis.set(xlabel=f"Mean {feature}", ylabel="AI - manual", title=f"{structure}: {feature}")
        figure.tight_layout()
        figure.savefig(args.output_dir / f"bland_altman_{structure}_{feature}.png", dpi=180)
        plt.close(figure)

    seed_offset = 0
    for structure in ("SSC", "HSC", "PSC"):
        block = paired.loc[paired["structure"] == structure]
        for feature in FEATURES:
            append_summary(block, structure, feature, seed_offset)
            seed_offset += 1
    for structure_a, structure_b in ANGLE_PAIRS:
        block = angles.loc[angles["structure"] == f"{structure_a}_{structure_b}"]
        append_summary(block, f"{structure_a}_{structure_b}", "inter_canal_angle_deg", seed_offset)
        seed_offset += 1

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "geometry_reliability_summary.csv", index=False, encoding="utf-8-sig")
    aggregate = {
        "status": "continuous_estimates_complete_gate_blocked" if gate["pass_threshold"] is None else "complete",
        "algorithm_version": GEOMETRY_ALGORITHM_VERSION,
        "people": int(paired["patient_key"].nunique()),
        "ears": int(len(paired) / 3),
        "mask_pairs": int(len(paired)),
        "bootstrap_repetitions": args.bootstrap,
        "bootstrap_unit": "patient",
        "seed": args.seed,
        "formal_pass_threshold": gate["pass_threshold"],
        "plane_angle_status": "complete_unsigned_physical_RAS_plus_PCA_normals",
        "summary": summary_rows,
    }
    (args.output_dir / "geometry_reliability_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in aggregate.items() if key != "summary"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
