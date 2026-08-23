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

from inner_ear_vit_seg3_experiment import CANAL_STRUCTS, scan_dataset
from inner_ear_vit_seg_experiment import (
    TinyViTUNet3D,
    bounding_box_center,
    crop_with_padding,
    load_nifti,
    normalize_intensity,
    resample_volume,
    resize_to_shape,
    train_subject_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transfer-train a patient-level three-channel 3D ViT canal segmenter."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("seg4"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--early-stop", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-size", nargs=3, type=int, default=(128, 128, 48))
    parser.add_argument("--target-spacing", nargs=3, type=float, default=(0.3472222, 0.3472222, 0.5))
    parser.add_argument(
        "--pretrained-union-checkpoint",
        type=Path,
        default=Path("analysis_out/vit_inner_ear_seg3_e45_c128/best_model.pt"),
    )
    parser.add_argument(
        "--centers-checkpoint",
        type=Path,
        default=Path(
            "results_md_progression/intermediate/semicircular_canal_vit_20260731/"
            "training/best_model.pt"
        ),
        help="Audited checkpoint containing training-only left/right crop centers.",
    )
    parser.add_argument("--head-lr-multiplier", type=float, default=2.0)
    parser.add_argument("--max-random-shift", type=int, default=4)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_mask(path: Path, reference_shape: tuple[int, int, int]) -> np.ndarray:
    data, _ = load_nifti(path)
    mask = (data > 0.5).astype(np.uint8)
    return (resize_to_shape(mask, reference_shape, order=0) > 0.5).astype(np.uint8)


def compute_centers(samples, target_spacing: tuple[float, float, float]) -> dict[str, np.ndarray]:
    centers: dict[str, list[np.ndarray]] = {"L": [], "R": []}
    for sample in samples:
        image, spacing = load_nifti(sample.image_path)
        masks = [load_mask(path, image.shape) for path in sample.mask_paths]
        union = np.logical_or.reduce(masks).astype(np.uint8)
        union = (resample_volume(union, spacing, target_spacing, order=0) > 0.5).astype(np.uint8)
        centers[sample.side].append(bounding_box_center(union))
    return {side: np.median(np.stack(values), axis=0).astype(np.float32) for side, values in centers.items()}


def prepare_crops(
    samples,
    crop_dir: Path,
    centers: dict[str, np.ndarray],
    target_spacing: tuple[float, float, float],
    crop_size: tuple[int, int, int],
    split_for_subject: dict[str, str],
) -> list[dict]:
    crop_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for index, sample in enumerate(samples, start=1):
        image_raw, spacing = load_nifti(sample.image_path)
        image = normalize_intensity(resample_volume(image_raw, spacing, target_spacing, order=1))
        masks = []
        coverages = []
        full_voxels = []
        crop_voxels = []
        for path in sample.mask_paths:
            mask = load_mask(path, image_raw.shape)
            mask = (resample_volume(mask, spacing, target_spacing, order=0) > 0.5).astype(np.uint8)
            cropped = crop_with_padding(mask, centers[sample.side], crop_size).astype(np.uint8)
            masks.append(cropped)
            full_n = int(mask.sum())
            crop_n = int(cropped.sum())
            full_voxels.append(full_n)
            crop_voxels.append(crop_n)
            coverages.append(0.0 if full_n == 0 else crop_n / full_n)
        cropped_image = crop_with_padding(image, centers[sample.side], crop_size).astype(np.float32)
        mask_stack = np.stack(masks, axis=0)
        save_path = crop_dir / f"{sample.sample_id}.npz"
        np.savez_compressed(save_path, image=cropped_image, mask=mask_stack)
        row = {
            "sample_id": sample.sample_id,
            "subject_id": sample.subject_id,
            "side": sample.side,
            "split": split_for_subject[sample.subject_id],
            "crop_path": str(save_path.resolve()),
        }
        for structure, full_n, crop_n, coverage in zip(
            CANAL_STRUCTS, full_voxels, crop_voxels, coverages, strict=True
        ):
            row[f"{structure}_full_voxels"] = full_n
            row[f"{structure}_crop_voxels"] = crop_n
            row[f"{structure}_coverage"] = coverage
        rows.append(row)
        if index % 25 == 0 or index == len(samples):
            print(f"CROP_PROGRESS {index}/{len(samples)}", flush=True)
    return rows


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


class CanalDataset(Dataset):
    def __init__(self, rows: list[dict], augment: bool = False, max_random_shift: int = 0):
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
            # Anatomical flips are intentionally excluded: for structure-specific labels,
            # superior/posterior orientation is part of the target identity.
            if self.max_random_shift > 0:
                shifts = tuple(
                    random.randint(-self.max_random_shift, self.max_random_shift) for _ in range(3)
                )
                image = translate_without_wrap(image, shifts, spatial_offset=0)
                mask = translate_without_wrap(mask, shifts, spatial_offset=1)
            image = image * (1.0 + random.uniform(-0.10, 0.10)) + random.uniform(-0.10, 0.10)
            image += np.random.normal(0.0, 0.03, image.shape).astype(np.float32)
        return torch.from_numpy(image[None]), torch.from_numpy(mask), row["sample_id"]


def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3, 4)
    intersection = torch.sum(probs * targets, dim=dims)
    denominator = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    soft_dice = (2.0 * intersection + 1e-5) / (denominator + 1e-5)

    false_positive = torch.sum(probs * (1.0 - targets), dim=dims)
    false_negative = torch.sum((1.0 - probs) * targets, dim=dims)
    tversky = (intersection + 1e-5) / (
        intersection + 0.65 * false_positive + 0.35 * false_negative + 1e-5
    )

    union_prob = 1.0 - torch.prod(1.0 - probs, dim=1, keepdim=True)
    union_target = torch.amax(targets, dim=1, keepdim=True)
    union_intersection = torch.sum(union_prob * union_target)
    union_dice = (2.0 * union_intersection + 1e-5) / (
        torch.sum(union_prob) + torch.sum(union_target) + 1e-5
    )

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability_of_target = targets * probs + (1.0 - targets) * (1.0 - probs)
    alpha = targets * 0.75 + (1.0 - targets) * 0.25
    focal = torch.mean(alpha * torch.pow(1.0 - probability_of_target, 2.0) * bce)

    return (
        0.55 * (1.0 - soft_dice.mean())
        + 0.15 * (1.0 - tversky.mean())
        + 0.15 * (1.0 - union_dice)
        + 0.15 * focal
    )


def initialise_from_union_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Pretrained union checkpoint not found: {checkpoint_path}")
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state = source.get("model_state", source)
    target_state = model.state_dict()
    loaded_keys: list[str] = []
    replicated_keys: list[str] = []
    for key, target_tensor in target_state.items():
        if key not in source_state:
            continue
        source_tensor = source_state[key]
        if source_tensor.shape == target_tensor.shape:
            target_state[key] = source_tensor.clone()
            loaded_keys.append(key)
        elif key == "head.weight" and source_tensor.shape[0] == 1 and target_tensor.shape[0] == len(CANAL_STRUCTS):
            target_state[key] = source_tensor.repeat(len(CANAL_STRUCTS), 1, 1, 1, 1)
            replicated_keys.append(key)
        elif key == "head.bias" and source_tensor.shape[0] == 1 and target_tensor.shape[0] == len(CANAL_STRUCTS):
            target_state[key] = source_tensor.repeat(len(CANAL_STRUCTS))
            replicated_keys.append(key)
    model.load_state_dict(target_state)
    return {
        "path": str(checkpoint_path.resolve()),
        "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "directly_loaded_tensor_count": len(loaded_keys),
        "replicated_head_tensor_count": len(replicated_keys),
        "total_tensor_count": len(target_state),
    }


def dice_by_channel(logits: torch.Tensor, targets: torch.Tensor, thresholds: list[float]) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    threshold_tensor = torch.tensor(thresholds, device=probs.device).view(1, -1, 1, 1, 1)
    pred = probs > threshold_tensor
    target = targets > 0.5
    dims = (2, 3, 4)
    inter = (pred & target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)
    return ((2.0 * inter + 1e-5) / (denom + 1e-5)).detach().cpu()


def run_epoch(model, loader, device, optimizer=None, scaler=None) -> tuple[float, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    dice_rows = []
    for image, mask, _ in loader:
        image = image.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(image)
            loss = loss_fn(logits, mask)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += float(loss.detach().cpu()) * image.shape[0]
        dice_rows.append(dice_by_channel(logits, mask, [0.5, 0.5, 0.5]).numpy())
    return total_loss / len(loader.dataset), np.concatenate(dice_rows, axis=0)


@torch.no_grad()
def tune_thresholds(model, loader, device) -> tuple[list[float], list[dict]]:
    model.eval()
    candidates = np.arange(0.30, 0.91, 0.05)
    score_grid = np.zeros((len(CANAL_STRUCTS), len(candidates)), dtype=float)
    count = 0
    for image, mask, _ in loader:
        image = image.to(device)
        mask = mask.to(device)
        logits = model(image)
        for candidate_index, candidate in enumerate(candidates):
            scores = dice_by_channel(logits, mask, [float(candidate)] * len(CANAL_STRUCTS)).numpy()
            score_grid[:, candidate_index] += scores.sum(axis=0)
        count += image.shape[0]
    score_grid /= max(count, 1)
    thresholds = [float(candidates[int(np.argmax(score_grid[channel]))]) for channel in range(len(CANAL_STRUCTS))]
    rows = [
        {
            "structure": structure,
            "threshold": float(candidate),
            "mean_validation_dice": float(score_grid[channel, candidate_index]),
        }
        for channel, structure in enumerate(CANAL_STRUCTS)
        for candidate_index, candidate in enumerate(candidates)
    ]
    return thresholds, rows


@torch.no_grad()
def evaluate(model, loader, device, thresholds: list[float], prediction_dir: Path | None = None):
    model.eval()
    rows = []
    total_loss = 0.0
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    for image, mask, sample_ids in loader:
        image_device = image.to(device)
        mask_device = mask.to(device)
        logits = model(image_device)
        total_loss += float(loss_fn(logits, mask_device).cpu()) * image.shape[0]
        probs = torch.sigmoid(logits)
        threshold_tensor = torch.tensor(thresholds, device=device).view(1, -1, 1, 1, 1)
        predictions = probs > threshold_tensor
        scores = dice_by_channel(logits, mask_device, thresholds).numpy()
        for batch_index, sample_id in enumerate(sample_ids):
            record = {"sample_id": sample_id}
            for channel, structure in enumerate(CANAL_STRUCTS):
                pred = predictions[batch_index, channel]
                target = mask_device[batch_index, channel] > 0.5
                intersection = int((pred & target).sum().cpu())
                pred_n = int(pred.sum().cpu())
                target_n = int(target.sum().cpu())
                union = pred_n + target_n - intersection
                record[f"{structure}_dice"] = float(scores[batch_index, channel])
                record[f"{structure}_iou"] = (intersection + 1e-5) / (union + 1e-5)
                record[f"{structure}_precision"] = (intersection + 1e-5) / (pred_n + 1e-5)
                record[f"{structure}_recall"] = (intersection + 1e-5) / (target_n + 1e-5)
            record["macro_dice"] = float(np.mean(scores[batch_index]))
            rows.append(record)
            if prediction_dir is not None:
                np.savez_compressed(
                    prediction_dir / f"{sample_id}.npz",
                    image=image[batch_index, 0].numpy().astype(np.float32),
                    target=mask[batch_index].numpy().astype(np.uint8),
                    prediction=predictions[batch_index].cpu().numpy().astype(np.uint8),
                    probability=probs[batch_index].cpu().numpy().astype(np.float16),
                )
    return total_loss / len(loader.dataset), rows


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = args.output_dir / "crops"
    samples, dataset_audit = scan_dataset(args.data_dir, CANAL_STRUCTS)
    subject_ids = sorted({sample.subject_id for sample in samples})
    train_subjects, val_subjects, test_subjects = train_subject_split(subject_ids, args.seed)
    split_for_subject = {
        subject_id: (
            "train" if subject_id in train_subjects else "validation" if subject_id in val_subjects else "test"
        )
        for subject_id in subject_ids
    }
    train_samples = [sample for sample in samples if sample.subject_id in train_subjects]
    metadata_path = args.output_dir / "sample_manifest.csv"
    cached_checkpoint_path = args.output_dir / "best_model.pt"
    cache_complete = metadata_path.exists() and len(list(crop_dir.glob("*.npz"))) == len(samples)
    if cache_complete and cached_checkpoint_path.exists():
        cached_checkpoint = torch.load(cached_checkpoint_path, map_location="cpu", weights_only=False)
        centers = {
            side: np.asarray(center, dtype=np.float32)
            for side, center in cached_checkpoint["side_centers"].items()
        }
    elif args.centers_checkpoint.exists():
        center_checkpoint = torch.load(args.centers_checkpoint, map_location="cpu", weights_only=False)
        centers = {
            side: np.asarray(center, dtype=np.float32)
            for side, center in center_checkpoint["side_centers"].items()
        }
        print(
            f"Reusing audited training-only crop centers from {args.centers_checkpoint}",
            flush=True,
        )
    else:
        centers = compute_centers(train_samples, tuple(args.target_spacing))
    if cache_complete:
        with metadata_path.open(encoding="utf-8-sig") as handle:
            metadata = list(csv.DictReader(handle))
        print("Reusing verified crop cache.", flush=True)
    else:
        metadata = prepare_crops(
            samples,
            crop_dir,
            centers,
            tuple(args.target_spacing),
            tuple(args.crop_size),
            split_for_subject,
        )
        write_csv(metadata_path, metadata)

    split_rows = {
        split: [row for row in metadata if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    loaders = {
        "train": DataLoader(
            CanalDataset(
                split_rows["train"],
                augment=True,
                max_random_shift=args.max_random_shift,
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        ),
        "validation": DataLoader(
            CanalDataset(split_rows["validation"]),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        ),
        "test": DataLoader(
            CanalDataset(split_rows["test"]),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        ),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyViTUNet3D(tuple(args.crop_size), out_channels=len(CANAL_STRUCTS))
    transfer_report = initialise_from_union_checkpoint(model, args.pretrained_union_checkpoint)
    model = model.to(device)
    head_parameters = list(model.head.parameters())
    head_parameter_ids = {id(parameter) for parameter in head_parameters}
    body_parameters = [parameter for parameter in model.parameters() if id(parameter) not in head_parameter_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": body_parameters, "lr": args.lr},
            {"params": head_parameters, "lr": args.lr * args.head_lr_multiplier},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    checkpoint_validation_loss = float("inf")
    best_validation_macro_dice = -1.0
    best_epoch = 0
    patience = 0
    history = []
    checkpoint_path = args.output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = run_epoch(model, loaders["train"], device, optimizer, scaler)
        with torch.no_grad():
            val_loss, val_dice = run_epoch(model, loaders["validation"], device)
        val_macro = float(val_dice.mean())
        scheduler.step(val_macro)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "train_macro_dice_at_0.5": float(train_dice.mean()),
            "validation_macro_dice_at_0.5": val_macro,
            **{
                f"validation_{structure}_dice_at_0.5": float(val_dice[:, channel].mean())
                for channel, structure in enumerate(CANAL_STRUCTS)
            },
            "body_learning_rate": optimizer.param_groups[0]["lr"],
            "head_learning_rate": optimizer.param_groups[1]["lr"],
        }
        history.append(row)
        write_csv(args.output_dir / "training_history.csv", history)
        print(
            f"EPOCH {epoch:02d}/{args.epochs} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_macro_dice={val_macro:.4f}",
            flush=True,
        )
        if val_macro > best_validation_macro_dice + 1e-4:
            best_validation_macro_dice = val_macro
            checkpoint_validation_loss = val_loss
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_loss": val_loss,
                    "validation_macro_dice_at_0.5": val_macro,
                    "transfer_initialisation": transfer_report,
                    "loss": {
                        "per_structure_soft_dice": 0.55,
                        "precision_weighted_tversky": 0.15,
                        "union_soft_dice": 0.15,
                        "focal_bce": 0.15,
                    },
                    "augmentation": {
                        "anatomical_flips": False,
                        "max_random_shift_voxels": args.max_random_shift,
                        "intensity_scale": 0.10,
                        "intensity_shift": 0.10,
                        "gaussian_noise_sd": 0.03,
                    },
                    "structures": list(CANAL_STRUCTS),
                    "crop_size": list(args.crop_size),
                    "target_spacing": list(args.target_spacing),
                    "side_centers": {side: center.tolist() for side, center in centers.items()},
                    "seed": args.seed,
                },
                checkpoint_path,
            )
        else:
            patience += 1
            if patience >= args.early_stop:
                print(f"EARLY_STOP epoch={epoch}", flush=True)
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    thresholds, threshold_rows = tune_thresholds(model, loaders["validation"], device)
    write_csv(args.output_dir / "validation_threshold_grid.csv", threshold_rows)
    test_loss, test_rows = evaluate(
        model,
        loaders["test"],
        device,
        thresholds,
        prediction_dir=args.output_dir / "test_predictions",
    )
    write_csv(args.output_dir / "internal_test_metrics.csv", test_rows)
    checkpoint["thresholds"] = thresholds
    torch.save(checkpoint, checkpoint_path)

    coverage_summary = {}
    for structure in CANAL_STRUCTS:
        values = np.asarray([float(row[f"{structure}_coverage"]) for row in metadata])
        coverage_summary[structure] = {
            "minimum": float(values.min()),
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "below_0.95_count": int((values < 0.95).sum()),
        }
    metric_summary = {
        structure: {
            metric: float(np.mean([float(row[f"{structure}_{metric}"]) for row in test_rows]))
            for metric in ("dice", "iou", "precision", "recall")
        }
        for structure in CANAL_STRUCTS
    }
    summary = {
        "device": str(device),
        "dataset_audit": dataset_audit,
        "patient_level_split": {
            "train_subjects": len(train_subjects),
            "validation_subjects": len(val_subjects),
            "test_subjects": len(test_subjects),
            "train_ears": len(split_rows["train"]),
            "validation_ears": len(split_rows["validation"]),
            "test_ears": len(split_rows["test"]),
            "seed": args.seed,
        },
        "structures": list(CANAL_STRUCTS),
        "best_epoch": best_epoch,
        "checkpoint_validation_loss": checkpoint_validation_loss,
        "best_validation_macro_dice_at_0.5": best_validation_macro_dice,
        "transfer_initialisation": transfer_report,
        "thresholds": dict(zip(CANAL_STRUCTS, thresholds, strict=True)),
        "test_loss": test_loss,
        "internal_test": metric_summary,
        "internal_test_macro_dice": float(np.mean([row["macro_dice"] for row in test_rows])),
        "side_centers_resampled_voxels": {side: center.tolist() for side, center in centers.items()},
        "crop_size": list(args.crop_size),
        "target_spacing_mm": list(args.target_spacing),
        "coverage_summary": coverage_summary,
        "external_validation_boundary": (
            "Zhejiang Second Hospital has no manual canal masks; external use is frozen inference "
            "plus anatomical/technical QC, not quantitative external segmentation validation."
        ),
    }
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("TRAINING_COMPLETE", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
