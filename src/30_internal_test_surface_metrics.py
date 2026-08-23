from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the deployed post-processing pipeline on the held-out test set.")
    parser.add_argument("--training-dir", type=Path, required=True)
    return parser.parse_args()


def retain_top_components(mask: np.ndarray, top_k: int) -> np.ndarray:
    if top_k == 0:
        return mask.astype(bool)
    labels, component_n = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if component_n <= top_k:
        return mask.astype(bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    selected = np.argsort(sizes)[-top_k:]
    return np.isin(labels, selected)


def surface_distances(pred: np.ndarray, target: np.ndarray, spacing: tuple[float, ...]) -> tuple[float, float]:
    if not pred.any() or not target.any():
        return float("nan"), float("nan")
    structure = np.ones((3, 3, 3), dtype=bool)
    pred_surface = pred ^ ndimage.binary_erosion(pred, structure=structure, border_value=0)
    target_surface = target ^ ndimage.binary_erosion(target, structure=structure, border_value=0)
    distance_to_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    distance_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate([distance_to_target[pred_surface], distance_to_pred[target_surface]])
    return float(np.percentile(distances, 95)), float(np.mean(distances))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.training_dir / "best_model.pt", map_location="cpu", weights_only=False)
    structures = checkpoint["structures"]
    thresholds = np.asarray(checkpoint["thresholds"], dtype=float)
    policies = checkpoint.get(
        "postprocess_policy",
        {
            structure: {
                "overlap_strategy": "argmax",
                "top_k_components": checkpoint.get("postprocess_top_k_components", {}).get(structure, 1),
            }
            for structure in structures
        },
    )
    spacing = tuple(float(value) for value in checkpoint["target_spacing"])
    voxel_volume = float(np.prod(spacing))
    rows = []
    for path in sorted((args.training_dir / "test_predictions").glob("*.npz")):
        with np.load(path) as data:
            target = data["target"].astype(bool)
            probability = data["probability"].astype(np.float32)
        raw_prediction = probability > thresholds[:, None, None, None]
        argmax_prediction = raw_prediction.copy()
        overlap = raw_prediction.sum(axis=0) > 1
        if overlap.any():
            winners = np.argmax(probability, axis=0)
            for channel in range(len(structures)):
                argmax_prediction[channel, overlap] = winners[overlap] == channel
        for channel, structure_name in enumerate(structures):
            policy = policies[structure_name]
            source = raw_prediction if policy["overlap_strategy"] == "none" else argmax_prediction
            pred = retain_top_components(
                source[channel],
                int(policy["top_k_components"]),
            )
            truth = target[channel]
            intersection = int((pred & truth).sum())
            pred_n = int(pred.sum())
            truth_n = int(truth.sum())
            union = pred_n + truth_n - intersection
            hd95, asd = surface_distances(pred, truth, spacing)
            truth_volume = truth_n * voxel_volume
            pred_volume = pred_n * voxel_volume
            rows.append(
                {
                    "sample_id": path.stem,
                    "structure": structure_name,
                    "dice": (2 * intersection + 1e-5) / (pred_n + truth_n + 1e-5),
                    "iou": (intersection + 1e-5) / (union + 1e-5),
                    "precision": (intersection + 1e-5) / (pred_n + 1e-5),
                    "recall": (intersection + 1e-5) / (truth_n + 1e-5),
                    "hd95_mm": hd95,
                    "average_symmetric_surface_distance_mm": asd,
                    "reference_volume_mm3": truth_volume,
                    "predicted_volume_mm3": pred_volume,
                    "absolute_volume_error_mm3": abs(pred_volume - truth_volume),
                    "relative_absolute_volume_error": (
                        abs(pred_volume - truth_volume) / truth_volume if truth_volume else None
                    ),
                    "overlap_voxels_resolved": int(overlap.sum()),
                }
            )
    write_csv(args.training_dir / "internal_test_deployment_metrics.csv", rows)
    summary = {}
    for structure in structures:
        subset = [row for row in rows if row["structure"] == structure]
        summary[structure] = {
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
    output = {
        "test_ear_count": len({row["sample_id"] for row in rows}),
        "structure_observation_count": len(rows),
        "postprocessing": {
            "per_structure_policy": policies,
            "selection_source": "validation only",
        },
        "per_structure_mean": summary,
        "macro_mean_dice": float(np.mean([value["dice"] for value in summary.values()])),
    }
    (args.training_dir / "internal_test_deployment_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
