from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


STRUCTURES = ("SSC", "HSC", "PSC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute held-out surface metrics for the frozen ensemble.")
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--spacing", nargs=3, type=float, default=(0.3472222, 0.3472222, 0.5))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def surface_distances(
    prediction: np.ndarray, target: np.ndarray, spacing: tuple[float, ...]
) -> tuple[float, float]:
    if not prediction.any() or not target.any():
        return float("nan"), float("nan")
    connectivity = np.ones((3, 3, 3), dtype=bool)
    prediction_surface = prediction ^ ndimage.binary_erosion(
        prediction, structure=connectivity, border_value=0
    )
    target_surface = target ^ ndimage.binary_erosion(target, structure=connectivity, border_value=0)
    distance_to_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    distance_to_prediction = ndimage.distance_transform_edt(~prediction_surface, sampling=spacing)
    distances = np.concatenate(
        [distance_to_target[prediction_surface], distance_to_prediction[target_surface]]
    )
    return float(np.percentile(distances, 95)), float(np.mean(distances))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean_ci(values: np.ndarray, repetitions: int, rng: np.random.Generator) -> list[float]:
    n = len(values)
    samples = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        samples[index] = float(np.mean(values[rng.integers(0, n, size=n)]))
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def main() -> None:
    args = parse_args()
    spacing = tuple(args.spacing)
    voxel_volume = float(np.prod(spacing))
    prediction_dir = args.training_dir / "test_predictions_postprocessed"
    rows: list[dict] = []
    macro_by_sample: dict[str, list[float]] = {}
    for path in sorted(prediction_dir.glob("*.npz")):
        with np.load(path) as data:
            prediction = data["prediction"].astype(bool)
            target = data["target"].astype(bool)
        macro_by_sample[path.stem] = []
        for channel, structure in enumerate(STRUCTURES):
            pred = prediction[channel]
            truth = target[channel]
            intersection = int((pred & truth).sum())
            pred_count = int(pred.sum())
            truth_count = int(truth.sum())
            union_count = pred_count + truth_count - intersection
            structure_dice = (2 * intersection + 1e-5) / (pred_count + truth_count + 1e-5)
            hd95, assd = surface_distances(pred, truth, spacing)
            macro_by_sample[path.stem].append(structure_dice)
            rows.append(
                {
                    "sample_id": path.stem,
                    "structure": structure,
                    "dice": structure_dice,
                    "iou": (intersection + 1e-5) / (union_count + 1e-5),
                    "precision": (intersection + 1e-5) / (pred_count + 1e-5),
                    "recall": (intersection + 1e-5) / (truth_count + 1e-5),
                    "hd95_mm": hd95,
                    "average_symmetric_surface_distance_mm": assd,
                    "reference_volume_mm3": truth_count * voxel_volume,
                    "predicted_volume_mm3": pred_count * voxel_volume,
                    "absolute_volume_error_mm3": abs(pred_count - truth_count) * voxel_volume,
                    "relative_absolute_volume_error": abs(pred_count - truth_count) / truth_count,
                }
            )
    write_csv(args.training_dir / "internal_test_surface_metrics.csv", rows)
    rng = np.random.default_rng(args.seed)
    per_structure = {}
    for structure in STRUCTURES:
        subset = [row for row in rows if row["structure"] == structure]
        dice_values = np.asarray([row["dice"] for row in subset], dtype=float)
        per_structure[structure] = {
            key: float(np.nanmean([row[key] for row in subset]))
            for key in (
                "dice",
                "iou",
                "precision",
                "recall",
                "hd95_mm",
                "average_symmetric_surface_distance_mm",
                "absolute_volume_error_mm3",
                "relative_absolute_volume_error",
            )
        }
        per_structure[structure]["dice_bootstrap_95_ci"] = bootstrap_mean_ci(
            dice_values, args.bootstrap, rng
        )
    macro_values = np.asarray(
        [np.mean(values) for values in macro_by_sample.values()], dtype=float
    )
    output = {
        "test_ear_count": len(macro_values),
        "selection_boundary": "All thresholds and component policies selected on validation only.",
        "per_structure_mean": per_structure,
        "macro_mean_dice": float(np.mean(macro_values)),
        "macro_dice_bootstrap_95_ci": bootstrap_mean_ci(macro_values, args.bootstrap, rng),
    }
    (args.training_dir / "internal_test_surface_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
