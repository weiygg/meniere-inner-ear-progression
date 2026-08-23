from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inner_ear_vit_seg3_experiment import CANAL_STRUCTS
from inner_ear_vit_seg_experiment import TinyViTUNet3D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train three independent binary 3D ViTs for SSC, HSC, and PSC."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrained-union-checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-size", nargs=3, type=int, default=(128, 128, 48))
    parser.add_argument("--target-spacing", nargs=3, type=float, default=(0.3472222, 0.3472222, 0.5))
    parser.add_argument("--max-random-shift", type=int, default=4)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def translate_without_wrap(array: np.ndarray, shifts: tuple[int, int, int], spatial_offset: int) -> np.ndarray:
    shifted = np.roll(array, shifts, axis=tuple(range(spatial_offset, spatial_offset + 3)))
    for spatial_axis, amount in enumerate(shifts):
        if amount == 0:
            continue
        axis = spatial_axis + spatial_offset
        slicer = [slice(None)] * shifted.ndim
        slicer[axis] = slice(0, amount) if amount > 0 else slice(amount, None)
        shifted[tuple(slicer)] = 0
    return shifted


class CropDataset(Dataset):
    def __init__(self, rows: list[dict], augment: bool, max_random_shift: int):
        self.rows = rows
        self.augment = augment
        self.max_random_shift = max_random_shift

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with np.load(row["crop_path"]) as data:
            image = data["image"].astype(np.float32)
            mask = data["mask"].astype(np.float32)
        if self.augment:
            shifts = tuple(
                random.randint(-self.max_random_shift, self.max_random_shift) for _ in range(3)
            )
            image = translate_without_wrap(image, shifts, spatial_offset=0)
            mask = translate_without_wrap(mask, shifts, spatial_offset=1)
            image = image * (1.0 + random.uniform(-0.10, 0.10)) + random.uniform(-0.10, 0.10)
            image += np.random.normal(0.0, 0.03, image.shape).astype(np.float32)
        return torch.from_numpy(image[None]), torch.from_numpy(mask), row["sample_id"]


def structure_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    dims = (2, 3, 4)
    true_positive = torch.sum(probability * target, dim=dims)
    false_positive = torch.sum(probability * (1.0 - target), dim=dims)
    false_negative = torch.sum((1.0 - probability) * target, dim=dims)
    dice = (2.0 * true_positive + 1e-5) / (
        2.0 * true_positive + false_positive + false_negative + 1e-5
    )
    tversky = (true_positive + 1e-5) / (
        true_positive + 0.65 * false_positive + 0.35 * false_negative + 1e-5
    )
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability_of_target = target * probability + (1.0 - target) * (1.0 - probability)
    alpha = target * 0.75 + (1.0 - target) * 0.25
    focal = torch.mean(alpha * torch.pow(1.0 - probability_of_target, 2.0) * bce)
    return 0.70 * (1.0 - dice.mean()) + 0.20 * (1.0 - tversky.mean()) + 0.10 * focal


def hard_dice(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    prediction = torch.sigmoid(logits) > threshold
    truth = target > 0.5
    dims = (2, 3, 4)
    intersection = (prediction & truth).sum(dim=dims)
    denominator = prediction.sum(dim=dims) + truth.sum(dim=dims)
    return ((2.0 * intersection + 1e-5) / (denominator + 1e-5)).detach().cpu().numpy()[:, 0]


def train_epoch(
    models: dict[str, torch.nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
    scalers: dict[str, torch.amp.GradScaler],
    loader: DataLoader,
    device: torch.device,
    active: dict[str, bool],
) -> dict[str, dict[str, float]]:
    totals = {name: {"loss": 0.0, "dice_sum": 0.0, "count": 0} for name in CANAL_STRUCTS}
    for model in models.values():
        model.train()
    for image, masks, _ in loader:
        image = image.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        for channel, structure in enumerate(CANAL_STRUCTS):
            if not active[structure]:
                continue
            optimizer = optimizers[structure]
            scaler = scalers[structure]
            optimizer.zero_grad(set_to_none=True)
            target = masks[:, channel : channel + 1]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = models[structure](image)
                loss = structure_loss(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(models[structure].parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            scores = hard_dice(logits, target)
            totals[structure]["loss"] += float(loss.detach().cpu()) * image.shape[0]
            totals[structure]["dice_sum"] += float(scores.sum())
            totals[structure]["count"] += image.shape[0]
    return {
        structure: {
            "loss": totals[structure]["loss"] / max(totals[structure]["count"], 1),
            "dice": totals[structure]["dice_sum"] / max(totals[structure]["count"], 1),
        }
        for structure in CANAL_STRUCTS
    }


@torch.no_grad()
def validate(
    models: dict[str, torch.nn.Module], loader: DataLoader, device: torch.device
) -> dict[str, dict[str, float]]:
    totals = {name: {"loss": 0.0, "dice_sum": 0.0, "count": 0} for name in CANAL_STRUCTS}
    for model in models.values():
        model.eval()
    for image, masks, _ in loader:
        image = image.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        for channel, structure in enumerate(CANAL_STRUCTS):
            target = masks[:, channel : channel + 1]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = models[structure](image)
                loss = structure_loss(logits, target)
            scores = hard_dice(logits, target)
            totals[structure]["loss"] += float(loss.cpu()) * image.shape[0]
            totals[structure]["dice_sum"] += float(scores.sum())
            totals[structure]["count"] += image.shape[0]
    return {
        structure: {
            "loss": totals[structure]["loss"] / totals[structure]["count"],
            "dice": totals[structure]["dice_sum"] / totals[structure]["count"],
        }
        for structure in CANAL_STRUCTS
    }


@torch.no_grad()
def tune_thresholds(
    models: dict[str, torch.nn.Module], loader: DataLoader, device: torch.device
) -> tuple[dict[str, float], list[dict]]:
    candidates = np.arange(0.20, 0.951, 0.05)
    score_sums = {structure: np.zeros(len(candidates), dtype=float) for structure in CANAL_STRUCTS}
    count = 0
    for model in models.values():
        model.eval()
    for image, masks, _ in loader:
        image = image.to(device)
        masks_np = masks.numpy().astype(bool)
        for channel, structure in enumerate(CANAL_STRUCTS):
            probability = torch.sigmoid(models[structure](image)).float().cpu().numpy()[:, 0]
            target = masks_np[:, channel]
            for index, threshold in enumerate(candidates):
                prediction = probability > threshold
                intersection = (prediction & target).reshape(len(image), -1).sum(axis=1)
                denominator = prediction.reshape(len(image), -1).sum(axis=1) + target.reshape(len(image), -1).sum(axis=1)
                score_sums[structure][index] += float(np.sum((2 * intersection + 1e-5) / (denominator + 1e-5)))
        count += len(image)
    rows: list[dict] = []
    selected: dict[str, float] = {}
    for structure in CANAL_STRUCTS:
        means = score_sums[structure] / count
        best_index = int(np.argmax(means))
        selected[structure] = float(candidates[best_index])
        for index, threshold in enumerate(candidates):
            rows.append(
                {
                    "structure": structure,
                    "threshold": float(threshold),
                    "mean_validation_dice": float(means[index]),
                }
            )
    return selected, rows


@torch.no_grad()
def evaluate_test(
    models: dict[str, torch.nn.Module],
    loader: DataLoader,
    device: torch.device,
    thresholds: dict[str, float],
    prediction_dir: Path,
) -> list[dict]:
    prediction_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for model in models.values():
        model.eval()
    for image, masks, sample_ids in loader:
        image_device = image.to(device)
        probabilities = {
            structure: torch.sigmoid(models[structure](image_device)).float().cpu().numpy()[:, 0]
            for structure in CANAL_STRUCTS
        }
        masks_np = masks.numpy().astype(bool)
        for sample_index, sample_id in enumerate(sample_ids):
            probability_stack = np.stack(
                [probabilities[structure][sample_index] for structure in CANAL_STRUCTS], axis=0
            )
            prediction_stack = np.stack(
                [
                    probabilities[structure][sample_index] > thresholds[structure]
                    for structure in CANAL_STRUCTS
                ],
                axis=0,
            )
            record = {"sample_id": sample_id}
            sample_dices: list[float] = []
            for channel, structure in enumerate(CANAL_STRUCTS):
                prediction = prediction_stack[channel]
                target = masks_np[sample_index, channel]
                intersection = int((prediction & target).sum())
                prediction_count = int(prediction.sum())
                target_count = int(target.sum())
                union_count = prediction_count + target_count - intersection
                dice = (2 * intersection + 1e-5) / (prediction_count + target_count + 1e-5)
                record[f"{structure}_dice"] = dice
                record[f"{structure}_iou"] = (intersection + 1e-5) / (union_count + 1e-5)
                record[f"{structure}_precision"] = (intersection + 1e-5) / (prediction_count + 1e-5)
                record[f"{structure}_recall"] = (intersection + 1e-5) / (target_count + 1e-5)
                sample_dices.append(dice)
            record["macro_dice"] = float(np.mean(sample_dices))
            rows.append(record)
            np.savez_compressed(
                prediction_dir / f"{sample_id}.npz",
                image=image[sample_index, 0].numpy().astype(np.float32),
                target=masks_np[sample_index].astype(np.uint8),
                prediction=prediction_stack.astype(np.uint8),
                probability=probability_stack.astype(np.float16),
            )
    return rows


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.manifest)
    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    loaders = {
        "train": DataLoader(
            CropDataset(split_rows["train"], augment=True, max_random_shift=args.max_random_shift),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        ),
        "validation": DataLoader(
            CropDataset(split_rows["validation"], augment=False, max_random_shift=0),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        ),
        "test": DataLoader(
            CropDataset(split_rows["test"], augment=False, max_random_shift=0),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        ),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_state = torch.load(args.pretrained_union_checkpoint, map_location="cpu", weights_only=False)
    source_state = source_state.get("model_state", source_state)
    models = {}
    optimizers = {}
    schedulers = {}
    scalers = {}
    for structure in CANAL_STRUCTS:
        model = TinyViTUNet3D(tuple(args.crop_size)).to(device)
        model.load_state_dict(source_state)
        models[structure] = model
        optimizers[structure] = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        schedulers[structure] = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizers[structure], mode="max", factor=0.5, patience=3
        )
        scalers[structure] = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_dice = {structure: -1.0 for structure in CANAL_STRUCTS}
    best_epoch = {structure: 0 for structure in CANAL_STRUCTS}
    patience = {structure: 0 for structure in CANAL_STRUCTS}
    active = {structure: True for structure in CANAL_STRUCTS}
    history: list[dict] = []
    source_sha256 = hashlib.sha256(args.pretrained_union_checkpoint.read_bytes()).hexdigest()

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(models, optimizers, scalers, loaders["train"], device, active)
        validation_metrics = validate(models, loaders["validation"], device)
        row: dict = {"epoch": epoch}
        for structure in CANAL_STRUCTS:
            schedulers[structure].step(validation_metrics[structure]["dice"])
            row[f"train_{structure}_loss"] = train_metrics[structure]["loss"]
            row[f"train_{structure}_dice_at_0.5"] = train_metrics[structure]["dice"]
            row[f"validation_{structure}_loss"] = validation_metrics[structure]["loss"]
            row[f"validation_{structure}_dice_at_0.5"] = validation_metrics[structure]["dice"]
            row[f"{structure}_learning_rate"] = optimizers[structure].param_groups[0]["lr"]
            row[f"{structure}_active"] = active[structure]
            if active[structure] and validation_metrics[structure]["dice"] > best_dice[structure] + 1e-4:
                best_dice[structure] = validation_metrics[structure]["dice"]
                best_epoch[structure] = epoch
                patience[structure] = 0
                torch.save(
                    {
                        "model_state": models[structure].state_dict(),
                        "structure": structure,
                        "epoch": epoch,
                        "validation_dice_at_0.5": best_dice[structure],
                        "crop_size": list(args.crop_size),
                        "target_spacing": list(args.target_spacing),
                        "seed": args.seed,
                        "pretrained_union_checkpoint": str(args.pretrained_union_checkpoint.resolve()),
                        "pretrained_union_sha256": source_sha256,
                        "roi_localisation": "Frozen union ViT predicted-center crop; no reference-mask centre at validation/test inference.",
                    },
                    args.output_dir / f"best_{structure}_model.pt",
                )
            elif active[structure]:
                patience[structure] += 1
                if patience[structure] >= args.early_stop:
                    active[structure] = False
                    print(f"STRUCTURE_EARLY_STOP {structure} epoch={epoch}", flush=True)
        row["validation_macro_dice_at_0.5"] = float(
            np.mean([validation_metrics[structure]["dice"] for structure in CANAL_STRUCTS])
        )
        history.append(row)
        write_csv(args.output_dir / "training_history.csv", history)
        print(
            f"EPOCH {epoch:02d}/{args.epochs} val_macro={row['validation_macro_dice_at_0.5']:.4f} "
            + " ".join(
                f"{structure}={validation_metrics[structure]['dice']:.4f}" for structure in CANAL_STRUCTS
            ),
            flush=True,
        )
        if not any(active.values()):
            break

    for structure in CANAL_STRUCTS:
        checkpoint = torch.load(
            args.output_dir / f"best_{structure}_model.pt", map_location=device, weights_only=False
        )
        models[structure].load_state_dict(checkpoint["model_state"])
    thresholds, threshold_rows = tune_thresholds(models, loaders["validation"], device)
    write_csv(args.output_dir / "validation_threshold_grid.csv", threshold_rows)
    test_rows = evaluate_test(
        models,
        loaders["test"],
        device,
        thresholds,
        args.output_dir / "test_predictions",
    )
    write_csv(args.output_dir / "internal_test_metrics.csv", test_rows)

    per_structure = {
        structure: {
            metric: float(np.mean([row[f"{structure}_{metric}"] for row in test_rows]))
            for metric in ("dice", "iou", "precision", "recall")
        }
        for structure in CANAL_STRUCTS
    }
    summary = {
        "device": str(device),
        "model_design": "Three independent binary 3D TinyViT-UNet models, one per canal structure.",
        "patient_level_split": {
            "train_ears": len(split_rows["train"]),
            "validation_ears": len(split_rows["validation"]),
            "test_ears": len(split_rows["test"]),
            "seed": args.seed,
        },
        "best_validation_dice_at_0.5": best_dice,
        "best_epoch": best_epoch,
        "thresholds_selected_on_validation": thresholds,
        "internal_test": per_structure,
        "internal_test_macro_dice": float(np.mean([row["macro_dice"] for row in test_rows])),
        "pretrained_union_checkpoint_sha256": source_sha256,
        "roi_localisation": "Frozen union ViT predicted-center crop; reference masks used only for QC/metrics.",
    }
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("ENSEMBLE_TRAINING_COMPLETE", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
