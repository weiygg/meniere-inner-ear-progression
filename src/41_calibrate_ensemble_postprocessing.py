from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select ensemble thresholds/components on validation only.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def load_training_module():
    path = Path(__file__).with_name("39_train_structure_specific_vit_ensemble.py")
    spec = importlib.util.spec_from_file_location("structure_ensemble_training", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def retain_top_components(mask: np.ndarray, top_k: int) -> np.ndarray:
    if top_k == 0:
        return mask.astype(bool)
    labels, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if count <= top_k:
        return mask.astype(bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    selected = np.argsort(sizes)[-top_k:]
    return np.isin(labels, selected)


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = int((prediction & target).sum())
    return (2 * intersection + 1e-5) / (int(prediction.sum()) + int(target.sum()) + 1e-5)


def main() -> None:
    args = parse_args()
    training = load_training_module()
    manifest = read_csv(args.manifest)
    validation_rows = [row for row in manifest if row["split"] == "validation"]
    validation_loader = DataLoader(
        training.CropDataset(validation_rows, augment=False, max_random_shift=0),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for structure in training.CANAL_STRUCTS:
        checkpoint = torch.load(
            args.training_dir / f"best_{structure}_model.pt", map_location=device, weights_only=False
        )
        model = training.TinyViTUNet3D(tuple(checkpoint["crop_size"])).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models[structure] = model

    probabilities = {structure: [] for structure in training.CANAL_STRUCTS}
    targets = {structure: [] for structure in training.CANAL_STRUCTS}
    with torch.no_grad():
        for image, masks, _ in validation_loader:
            image = image.to(device)
            for channel, structure in enumerate(training.CANAL_STRUCTS):
                probability = torch.sigmoid(models[structure](image)).float().cpu().numpy()[:, 0]
                probabilities[structure].extend(probability)
                targets[structure].extend(masks.numpy()[:, channel].astype(bool))

    threshold_candidates = np.arange(0.10, 0.901, 0.05)
    component_candidates = (0, 1, 2, 3)
    grid_rows: list[dict] = []
    policies: dict[str, dict] = {}
    for structure in training.CANAL_STRUCTS:
        best = None
        for threshold in threshold_candidates:
            for top_k in component_candidates:
                scores = [
                    dice(retain_top_components(probability > threshold, top_k), target)
                    for probability, target in zip(
                        probabilities[structure], targets[structure], strict=True
                    )
                ]
                row = {
                    "structure": structure,
                    "threshold": float(threshold),
                    "top_k_components": top_k,
                    "mean_validation_dice": float(np.mean(scores)),
                }
                grid_rows.append(row)
                if best is None or row["mean_validation_dice"] > best["mean_validation_dice"]:
                    best = row
        assert best is not None
        policies[structure] = {
            "threshold": best["threshold"],
            "top_k_components": best["top_k_components"],
            "validation_mean_dice": best["mean_validation_dice"],
        }
    training.write_csv(args.training_dir / "validation_postprocessing_grid.csv", grid_rows)

    output_prediction_dir = args.training_dir / "test_predictions_postprocessed"
    output_prediction_dir.mkdir(parents=True, exist_ok=True)
    test_rows: list[dict] = []
    for path in sorted((args.training_dir / "test_predictions").glob("*.npz")):
        with np.load(path) as data:
            image = data["image"].astype(np.float32)
            target = data["target"].astype(bool)
            probability = data["probability"].astype(np.float32)
        prediction = np.zeros_like(target, dtype=bool)
        record = {"sample_id": path.stem}
        sample_dices = []
        for channel, structure in enumerate(training.CANAL_STRUCTS):
            policy = policies[structure]
            prediction[channel] = retain_top_components(
                probability[channel] > policy["threshold"],
                int(policy["top_k_components"]),
            )
            pred = prediction[channel]
            truth = target[channel]
            intersection = int((pred & truth).sum())
            pred_count = int(pred.sum())
            truth_count = int(truth.sum())
            union_count = pred_count + truth_count - intersection
            structure_dice = (2 * intersection + 1e-5) / (pred_count + truth_count + 1e-5)
            record[f"{structure}_dice"] = structure_dice
            record[f"{structure}_iou"] = (intersection + 1e-5) / (union_count + 1e-5)
            record[f"{structure}_precision"] = (intersection + 1e-5) / (pred_count + 1e-5)
            record[f"{structure}_recall"] = (intersection + 1e-5) / (truth_count + 1e-5)
            sample_dices.append(structure_dice)
        record["macro_dice"] = float(np.mean(sample_dices))
        test_rows.append(record)
        np.savez_compressed(
            output_prediction_dir / path.name,
            image=image,
            target=target.astype(np.uint8),
            prediction=prediction.astype(np.uint8),
            probability=probability.astype(np.float16),
        )
    training.write_csv(args.training_dir / "internal_test_postprocessed_metrics.csv", test_rows)
    per_structure = {
        structure: {
            metric: float(np.mean([row[f"{structure}_{metric}"] for row in test_rows]))
            for metric in ("dice", "iou", "precision", "recall")
        }
        for structure in training.CANAL_STRUCTS
    }
    output = {
        "selection_source": "validation only",
        "policies": policies,
        "internal_test": per_structure,
        "internal_test_macro_dice": float(np.mean([row["macro_dice"] for row in test_rows])),
    }
    (args.training_dir / "postprocessed_metrics_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics_path = args.training_dir / "metrics_summary.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["deployed_postprocessing"] = policies
    metrics["internal_test_postprocessed"] = per_structure
    metrics["internal_test_postprocessed_macro_dice"] = output["internal_test_macro_dice"]
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
