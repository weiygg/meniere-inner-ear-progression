from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare label-free two-stage ViT ROI crops for three-canal training."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("seg4"))
    parser.add_argument("--fixed-crop-dir", type=Path, required=True)
    parser.add_argument("--fixed-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--union-checkpoint", type=Path, required=True)
    parser.add_argument("--centers-checkpoint", type=Path, required=True)
    parser.add_argument("--crop-size", nargs=3, type=int, default=(128, 128, 48))
    parser.add_argument("--target-spacing", nargs=3, type=float, default=(0.3472222, 0.3472222, 0.5))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--refinement-passes", type=int, default=1)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_mask(path: Path, reference_shape: tuple[int, int, int]) -> np.ndarray:
    data, _ = load_nifti(path)
    mask = (data > 0.5).astype(np.uint8)
    return (resize_to_shape(mask, reference_shape, order=0) > 0.5).astype(np.uint8)


def retain_largest_components(mask: np.ndarray, top_k: int = 3) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if count <= top_k:
        return mask.astype(bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    selected = np.argsort(sizes)[-top_k:]
    return np.isin(labels, selected)


@torch.no_grad()
def predict_local_center(
    model: torch.nn.Module,
    image_crop: np.ndarray,
    threshold: float,
    device: torch.device,
) -> tuple[np.ndarray | None, int, float]:
    image_tensor = torch.from_numpy(image_crop[None, None].astype(np.float32)).to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        probability = torch.sigmoid(model(image_tensor))[0, 0]
    probability_np = probability.float().cpu().numpy()
    prediction = retain_largest_components(probability_np > threshold, top_k=3)
    if not prediction.any():
        return None, 0, float(probability_np.max())
    return bounding_box_center(prediction), int(prediction.sum()), float(probability_np.max())


def local_to_global_center(
    local_center: np.ndarray,
    fixed_center: np.ndarray,
    crop_size: tuple[int, int, int],
    full_shape: tuple[int, int, int],
) -> np.ndarray:
    global_center = []
    for axis, size in enumerate(crop_size):
        raw_start = int(round(float(fixed_center[axis]) - size / 2))
        pad_before = max(0, -raw_start)
        actual_start = max(0, raw_start)
        coordinate = actual_start + float(local_center[axis]) - pad_before
        global_center.append(np.clip(coordinate, 0, full_shape[axis] - 1))
    return np.asarray(global_center, dtype=np.float32)


def refine_global_center(
    local_center: np.ndarray,
    current_global_center: np.ndarray,
    crop_size: tuple[int, int, int],
    full_shape: tuple[int, int, int],
) -> np.ndarray:
    desired_local_center = np.asarray(crop_size, dtype=np.float32) / 2.0
    refined = current_global_center + local_center - desired_local_center
    return np.clip(refined, 0, np.asarray(full_shape, dtype=np.float32) - 1).astype(np.float32)


def main() -> None:
    args = parse_args()
    crop_size = tuple(args.crop_size)
    target_spacing = tuple(args.target_spacing)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    crop_output = args.output_dir / "crops"
    crop_output.mkdir(parents=True, exist_ok=True)

    fixed_rows = {row["sample_id"]: row for row in read_csv(args.fixed_manifest)}
    samples, _ = scan_dataset(args.data_dir, CANAL_STRUCTS)
    center_checkpoint = torch.load(args.centers_checkpoint, map_location="cpu", weights_only=False)
    fixed_centers = {
        side: np.asarray(center, dtype=np.float32)
        for side, center in center_checkpoint["side_centers"].items()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyViTUNet3D(crop_size).to(device)
    union_state = torch.load(args.union_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(union_state.get("model_state", union_state))
    model.eval()

    rows: list[dict] = []
    for index, sample in enumerate(samples, start=1):
        fixed_crop_path = args.fixed_crop_dir / f"{sample.sample_id}.npz"
        with np.load(fixed_crop_path) as fixed_crop:
            fixed_image = fixed_crop["image"].astype(np.float32)
        first_local_center, first_voxels, first_max_probability = predict_local_center(
            model, fixed_image, args.threshold, device
        )

        raw_image, spacing = load_nifti(sample.image_path)
        full_image = normalize_intensity(resample_volume(raw_image, spacing, target_spacing, order=1))
        full_shape = full_image.shape
        fallback = first_local_center is None
        if first_local_center is None:
            predicted_global_center = fixed_centers[sample.side].copy()
        else:
            predicted_global_center = local_to_global_center(
                first_local_center,
                fixed_centers[sample.side],
                crop_size,
                full_shape,
            )

        refinement_successes = 0
        for _ in range(args.refinement_passes):
            candidate_image = crop_with_padding(full_image, predicted_global_center, crop_size).astype(np.float32)
            local_center, _, _ = predict_local_center(model, candidate_image, args.threshold, device)
            if local_center is None:
                break
            predicted_global_center = refine_global_center(
                local_center, predicted_global_center, crop_size, full_shape
            )
            refinement_successes += 1

        final_image = crop_with_padding(full_image, predicted_global_center, crop_size).astype(np.float32)
        masks: list[np.ndarray] = []
        full_voxels: list[int] = []
        crop_voxels: list[int] = []
        coverages: list[float] = []
        for mask_path in sample.mask_paths:
            mask = load_mask(mask_path, raw_image.shape)
            mask = (resample_volume(mask, spacing, target_spacing, order=0) > 0.5).astype(np.uint8)
            cropped_mask = crop_with_padding(mask, predicted_global_center, crop_size).astype(np.uint8)
            full_count = int(mask.sum())
            crop_count = int(cropped_mask.sum())
            masks.append(cropped_mask)
            full_voxels.append(full_count)
            crop_voxels.append(crop_count)
            coverages.append(0.0 if full_count == 0 else crop_count / full_count)

        save_path = crop_output / f"{sample.sample_id}.npz"
        np.savez_compressed(save_path, image=final_image, mask=np.stack(masks, axis=0))
        fixed_row = fixed_rows[sample.sample_id]
        row = {
            "sample_id": sample.sample_id,
            "subject_id": sample.subject_id,
            "side": sample.side,
            "split": fixed_row["split"],
            "crop_path": str(save_path.resolve()),
            "localizer_fallback_to_fixed_center": fallback,
            "localizer_first_prediction_voxels": first_voxels,
            "localizer_first_max_probability": first_max_probability,
            "localizer_refinement_successes": refinement_successes,
            "predicted_center_x": float(predicted_global_center[0]),
            "predicted_center_y": float(predicted_global_center[1]),
            "predicted_center_z": float(predicted_global_center[2]),
        }
        for structure, full_count, crop_count, coverage in zip(
            CANAL_STRUCTS, full_voxels, crop_voxels, coverages, strict=True
        ):
            row[f"{structure}_full_voxels"] = full_count
            row[f"{structure}_crop_voxels"] = crop_count
            row[f"{structure}_coverage"] = coverage
        rows.append(row)
        if index % 10 == 0 or index == len(samples):
            print(f"PREDICTED_ROI_PROGRESS {index}/{len(samples)}", flush=True)

    write_csv(args.output_dir / "sample_manifest.csv", rows)
    print("PREDICTED_ROI_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
