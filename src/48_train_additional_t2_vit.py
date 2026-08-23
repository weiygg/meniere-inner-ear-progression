from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inner_ear_vit_seg_experiment import TinyViTUNet3D


STRUCTURES = ("Cochlear", "Vestibular", "TV")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train independent transfer-learned 3D ViTs for additional T2 masks.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results_md_progression/intermediate/all_t2_vit_20260801/additional_training_crops/sample_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_md_progression/final/all_t2_inner_ear_vit_20260801/additional_models"),
    )
    parser.add_argument(
        "--pretrained-union-checkpoint",
        type=Path,
        default=Path(
            "results_md_progression/final/semicircular_canal_vit_20260731/"
            "model_v2_structure_ensemble/best_SSC_model.pt"
        ),
    )
    parser.add_argument("--structures", default=",".join(STRUCTURES))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--early-stop", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-size", nargs=3, type=int, default=(128, 128, 48))
    parser.add_argument("--target-spacing", nargs=3, type=float, default=(0.3472222, 0.3472222, 0.5))
    parser.add_argument("--max-random-shift", type=int, default=4)
    parser.add_argument(
        "--reuse-checkpoints",
        action="store_true",
        help="Skip fitting for a requested structure when its best checkpoint already exists; still tune and test it.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


class SingleStructureDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], augment: bool, max_random_shift: int):
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
            shifts = tuple(random.randint(-self.max_random_shift, self.max_random_shift) for _ in range(3))
            image = translate_without_wrap(image, shifts, spatial_offset=0)
            mask = translate_without_wrap(mask, shifts, spatial_offset=1)
            image = image * (1.0 + random.uniform(-0.10, 0.10)) + random.uniform(-0.10, 0.10)
            image += np.random.normal(0.0, 0.03, image.shape).astype(np.float32)
        return torch.from_numpy(image[None]), torch.from_numpy(mask), row["sample_id"], row["subject_id"]


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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


def batch_dice(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    prediction = torch.sigmoid(logits) > threshold
    truth = target > 0.5
    intersection = (prediction & truth).sum(dim=(2, 3, 4))
    denominator = prediction.sum(dim=(2, 3, 4)) + truth.sum(dim=(2, 3, 4))
    return ((2.0 * intersection + 1e-5) / (denominator + 1e-5)).detach().cpu().numpy()[:, 0]


def retain_components(mask: np.ndarray, top_k: int) -> np.ndarray:
    if top_k == 0:
        return mask.astype(bool)
    labels, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if count <= top_k:
        return mask.astype(bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    selected = np.argsort(sizes)[-top_k:]
    return np.isin(labels, selected)


def dice_value(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = int((prediction & target).sum())
    return (2 * intersection + 1e-5) / (int(prediction.sum()) + int(target.sum()) + 1e-5)


def make_loader(rows: list[dict[str, str]], args: argparse.Namespace, augment: bool, shuffle: bool) -> DataLoader:
    return DataLoader(
        SingleStructureDataset(rows, augment=augment, max_random_shift=args.max_random_shift if augment else 0),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )


def train_one_epoch(model, loader, optimizer, scaler, device) -> dict[str, float]:
    model.train()
    loss_sum = dice_sum = 0.0
    count = 0
    for image, target, _, _ in loader:
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(image)
            loss = segmentation_loss(logits, target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        scores = batch_dice(logits, target)
        loss_sum += float(loss.detach().cpu()) * len(image)
        dice_sum += float(scores.sum())
        count += len(image)
    return {"loss": loss_sum / count, "dice": dice_sum / count}


@torch.no_grad()
def validate(model, loader, device) -> dict[str, float]:
    model.eval()
    loss_sum = dice_sum = 0.0
    count = 0
    for image, target, _, _ in loader:
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(image)
            loss = segmentation_loss(logits, target)
        scores = batch_dice(logits, target)
        loss_sum += float(loss.cpu()) * len(image)
        dice_sum += float(scores.sum())
        count += len(image)
    return {"loss": loss_sum / count, "dice": dice_sum / count}


@torch.no_grad()
def collect_probabilities(model, loader, device) -> list[dict[str, object]]:
    model.eval()
    rows: list[dict[str, object]] = []
    for image, target, sample_ids, subject_ids in loader:
        probability = torch.sigmoid(model(image.to(device))).float().cpu().numpy()[:, 0]
        truth = target.numpy().astype(bool)[:, 0]
        for index, sample_id in enumerate(sample_ids):
            rows.append(
                {
                    "sample_id": sample_id,
                    "subject_id": subject_ids[index],
                    "probability": probability[index],
                    "target": truth[index],
                    "image": image[index, 0].numpy().astype(np.float32),
                }
            )
    return rows


def tune_postprocessing(rows: list[dict[str, object]]) -> tuple[dict[str, object], list[dict[str, object]]]:
    candidates = np.arange(0.05, 0.951, 0.05)
    grid: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for threshold in candidates:
        for top_k in (0, 1, 2):
            scores = [
                dice_value(retain_components(row["probability"] > threshold, top_k), row["target"])
                for row in rows
            ]
            current = {"threshold": float(threshold), "top_k_components": top_k, "mean_validation_dice": float(np.mean(scores))}
            grid.append(current)
            if best is None or float(current["mean_validation_dice"]) > float(best["mean_validation_dice"]):
                best = current
    assert best is not None
    return best, grid


def bootstrap_patient_ci(rows: list[dict[str, object]], seed: int, n_boot: int = 2000) -> tuple[float, float]:
    by_subject: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_subject[str(row["subject_id"])].append(float(row["dice"]))
    subject_means = np.asarray([np.mean(values) for values in by_subject.values()], dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.asarray(
        [np.mean(rng.choice(subject_means, size=len(subject_means), replace=True)) for _ in range(n_boot)]
    )
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def evaluate_and_save(
    rows: list[dict[str, object]], postprocess: dict[str, object], output_dir: Path, seed: int
) -> tuple[list[dict[str, object]], dict[str, object]]:
    threshold = float(postprocess["threshold"])
    top_k = int(postprocess["top_k_components"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, object]] = []
    for row in rows:
        prediction = retain_components(row["probability"] > threshold, top_k)
        target = row["target"]
        intersection = int((prediction & target).sum())
        prediction_count = int(prediction.sum())
        target_count = int(target.sum())
        union_count = prediction_count + target_count - intersection
        metric_row = {
            "sample_id": row["sample_id"],
            "subject_id": row["subject_id"],
            "dice": (2 * intersection + 1e-5) / (prediction_count + target_count + 1e-5),
            "iou": (intersection + 1e-5) / (union_count + 1e-5),
            "precision": (intersection + 1e-5) / (prediction_count + 1e-5),
            "recall": (intersection + 1e-5) / (target_count + 1e-5),
            "prediction_voxels": prediction_count,
            "target_voxels": target_count,
        }
        metrics.append(metric_row)
        np.savez_compressed(
            output_dir / f"{row['sample_id']}.npz",
            image=row["image"],
            target=target.astype(np.uint8),
            prediction=prediction.astype(np.uint8),
            probability=row["probability"].astype(np.float16),
        )
    low, high = bootstrap_patient_ci(metrics, seed)
    summary = {
        metric: float(np.mean([row[metric] for row in metrics]))
        for metric in ("dice", "iou", "precision", "recall")
    }
    summary["dice_patient_cluster_bootstrap_95_ci"] = [low, high]
    summary["ears"] = len(metrics)
    summary["patients"] = len({row["subject_id"] for row in metrics})
    return metrics, summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = read_csv(args.manifest)
    requested = tuple(item.strip() for item in args.structures.split(",") if item.strip())
    invalid = set(requested) - set(STRUCTURES)
    if invalid:
        raise ValueError(f"Unsupported structures: {sorted(invalid)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_checkpoint = torch.load(args.pretrained_union_checkpoint, map_location="cpu", weights_only=False)
    source_state = source_checkpoint.get("model_state", source_checkpoint)
    source_sha256 = hashlib.sha256(args.pretrained_union_checkpoint.read_bytes()).hexdigest()
    summary_path = args.output_dir / "metrics_summary.json"
    final_summary: dict[str, object] = {
        "device": str(device),
        "model_design": "Independent binary 3D TinyViT-UNet models initialized from the established 48-slice SSC ViT.",
        "pretrained_union_sha256": source_sha256,
        "patient_level_split_reused_from_frozen_canal_development": True,
        "structures": {},
    }
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(previous.get("structures"), dict):
            final_summary["structures"].update(previous["structures"])

    for structure_index, structure in enumerate(requested):
        set_seed(args.seed + structure_index)
        structure_rows = [row for row in all_rows if row["structure"] == structure]
        split_rows = {
            split: [row for row in structure_rows if row["split"] == split]
            for split in ("train", "validation", "test")
        }
        if any(not split_rows[split] for split in split_rows):
            raise RuntimeError(f"{structure} has an empty split: { {k: len(v) for k, v in split_rows.items()} }")
        train_loader = make_loader(split_rows["train"], args, augment=True, shuffle=True)
        validation_loader = make_loader(split_rows["validation"], args, augment=False, shuffle=False)
        test_loader = make_loader(split_rows["test"], args, augment=False, shuffle=False)

        model = TinyViTUNet3D(tuple(args.crop_size)).to(device)
        model.load_state_dict(source_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        best_dice = -1.0
        best_epoch = 0
        patience = 0
        history: list[dict[str, object]] = []
        checkpoint_path = args.output_dir / f"best_{structure}_model.pt"
        if args.reuse_checkpoints and checkpoint_path.exists():
            existing = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            best_dice = float(existing.get("validation_dice_at_0.5", float("nan")))
            best_epoch = int(existing.get("epoch", 0))
            print(f"REUSE_CHECKPOINT {structure} epoch={best_epoch} val={best_dice:.4f}", flush=True)
        else:
            for epoch in range(1, args.epochs + 1):
                train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device)
                validation_metrics = validate(model, validation_loader, device)
                scheduler.step(validation_metrics["dice"])
                history.append(
                    {
                        "structure": structure,
                        "epoch": epoch,
                        "train_loss": train_metrics["loss"],
                        "train_dice_at_0.5": train_metrics["dice"],
                        "validation_loss": validation_metrics["loss"],
                        "validation_dice_at_0.5": validation_metrics["dice"],
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )
                write_csv(args.output_dir / f"training_history_{structure}.csv", history)
                print(
                    f"EPOCH {structure} {epoch:02d}/{args.epochs} train={train_metrics['dice']:.4f} val={validation_metrics['dice']:.4f}",
                    flush=True,
                )
                if validation_metrics["dice"] > best_dice + 1e-4:
                    best_dice = validation_metrics["dice"]
                    best_epoch = epoch
                    patience = 0
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "structure": structure,
                            "epoch": epoch,
                            "validation_dice_at_0.5": best_dice,
                            "crop_size": list(args.crop_size),
                            "target_spacing": list(args.target_spacing),
                            "seed": args.seed + structure_index,
                            "pretrained_union_checkpoint": str(args.pretrained_union_checkpoint.resolve()),
                            "pretrained_union_sha256": source_sha256,
                            "roi_localisation": "Frozen union ViT predicted-center crop.",
                        },
                        checkpoint_path,
                    )
                else:
                    patience += 1
                    if patience >= args.early_stop:
                        break

        best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model_state"])
        validation_predictions = collect_probabilities(model, validation_loader, device)
        selected, grid = tune_postprocessing(validation_predictions)
        write_csv(args.output_dir / f"validation_postprocessing_grid_{structure}.csv", grid)
        test_predictions = collect_probabilities(model, test_loader, device)
        metrics, test_summary = evaluate_and_save(
            test_predictions,
            selected,
            args.output_dir / "test_predictions" / structure,
            seed=args.seed + 1000 + structure_index,
        )
        write_csv(args.output_dir / f"internal_test_metrics_{structure}.csv", metrics)
        final_summary["structures"][structure] = {
            "labelled_ears": len(structure_rows),
            "labelled_patients": len({row["subject_id"] for row in structure_rows}),
            "split_ears": {split: len(rows) for split, rows in split_rows.items()},
            "best_epoch": best_epoch,
            "best_validation_dice_at_0.5": best_dice,
            "selected_postprocessing": selected,
            "internal_test": test_summary,
            "limitation": "Preliminary internal estimate; small labelled cohort, especially validation/test for Cochlear and Vestibular.",
        }
        summary_path.write_text(
            json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("ADDITIONAL_T2_TRAINING_COMPLETE", json.dumps(final_summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
