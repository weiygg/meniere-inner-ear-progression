from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze, calibrate, and test the structure-specific ViT ensemble.")
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


def main() -> None:
    args = parse_args()
    training = load_training_module()
    rows = read_csv(args.manifest)
    validation_rows = [row for row in rows if row["split"] == "validation"]
    test_rows_manifest = [row for row in rows if row["split"] == "test"]
    validation_loader = DataLoader(
        training.CropDataset(validation_rows, augment=False, max_random_shift=0),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        training.CropDataset(test_rows_manifest, augment=False, max_random_shift=0),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    checkpoints = {}
    for structure in training.CANAL_STRUCTS:
        checkpoint_path = args.training_dir / f"best_{structure}_model.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = training.TinyViTUNet3D(tuple(checkpoint["crop_size"])).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models[structure] = model
        checkpoints[structure] = checkpoint

    thresholds, threshold_rows = training.tune_thresholds(models, validation_loader, device)
    training.write_csv(args.training_dir / "validation_threshold_grid.csv", threshold_rows)
    test_rows = training.evaluate_test(
        models,
        test_loader,
        device,
        thresholds,
        args.training_dir / "test_predictions",
    )
    training.write_csv(args.training_dir / "internal_test_metrics.csv", test_rows)

    for structure in training.CANAL_STRUCTS:
        checkpoints[structure]["threshold_selected_on_validation"] = thresholds[structure]
        torch.save(checkpoints[structure], args.training_dir / f"best_{structure}_model.pt")

    per_structure = {
        structure: {
            metric: float(np.mean([float(row[f"{structure}_{metric}"]) for row in test_rows]))
            for metric in ("dice", "iou", "precision", "recall")
        }
        for structure in training.CANAL_STRUCTS
    }
    union_dices = []
    for prediction_path in sorted((args.training_dir / "test_predictions").glob("*.npz")):
        with np.load(prediction_path) as data:
            prediction_union = data["prediction"].astype(bool).any(axis=0)
            target_union = data["target"].astype(bool).any(axis=0)
        intersection = int((prediction_union & target_union).sum())
        union_dices.append(
            (2 * intersection + 1e-5)
            / (int(prediction_union.sum()) + int(target_union.sum()) + 1e-5)
        )

    summary = {
        "device": str(device),
        "model_design": "Three independent binary 3D TinyViT-UNet models, one per canal structure.",
        "patient_level_split": {
            "train_ears": sum(row["split"] == "train" for row in rows),
            "validation_ears": len(validation_rows),
            "test_ears": len(test_rows_manifest),
            "seed": 42,
        },
        "best_validation_dice_at_0.5": {
            structure: float(checkpoints[structure]["validation_dice_at_0.5"])
            for structure in training.CANAL_STRUCTS
        },
        "best_epoch": {
            structure: int(checkpoints[structure]["epoch"])
            for structure in training.CANAL_STRUCTS
        },
        "thresholds_selected_on_validation": thresholds,
        "internal_test": per_structure,
        "internal_test_macro_dice": float(
            np.mean([float(row["macro_dice"]) for row in test_rows])
        ),
        "internal_test_union_dice_secondary": float(np.mean(union_dices)),
        "roi_localisation": "Frozen union ViT predicted-center crop; reference masks used only for QC and metrics.",
        "external_validation_boundary": "Zhejiang Second Hospital lacks manual reference masks; external quantitative Dice remains unavailable.",
    }
    (args.training_dir / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("ENSEMBLE_FINALIZATION_COMPLETE", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
