from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inner_ear_vit_seg_experiment import (
    TinyViTUNet3D,
    crop_with_padding,
    load_nifti,
    normalize_intensity,
    resample_volume,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen external three-canal ViT inference.")
    parser.add_argument("--nifti-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def crop_bounds(
    full_shape: tuple[int, int, int],
    center: np.ndarray,
    crop_size: tuple[int, int, int],
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    full_slices = []
    crop_slices = []
    for dim, size in enumerate(crop_size):
        raw_start = int(round(float(center[dim]) - size / 2))
        raw_end = raw_start + size
        full_start = max(0, raw_start)
        full_end = min(full_shape[dim], raw_end)
        crop_start = full_start - raw_start
        crop_end = crop_start + (full_end - full_start)
        full_slices.append(slice(full_start, full_end))
        crop_slices.append(slice(crop_start, crop_end))
    return tuple(full_slices), tuple(crop_slices)


def retain_top_components(mask: np.ndarray, top_k: int) -> tuple[np.ndarray, int]:
    labels, component_n = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if component_n == 0 or top_k == 0 or component_n <= top_k:
        return mask.astype(np.uint8), component_n
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    selected = np.argsort(sizes)[-top_k:]
    return np.isin(labels, selected).astype(np.uint8), component_n


def boundary_touch(mask: np.ndarray) -> bool:
    if not mask.any():
        return False
    return bool(
        mask[0].any()
        or mask[-1].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
        or mask[:, :, 0].any()
        or mask[:, :, -1].any()
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_root = args.output_dir / "predicted_masks"
    mask_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    structures = tuple(checkpoint["structures"])
    crop_size = tuple(int(v) for v in checkpoint["crop_size"])
    target_spacing = tuple(float(v) for v in checkpoint["target_spacing"])
    thresholds = [float(v) for v in checkpoint["thresholds"]]
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
    reference_shape = np.asarray(checkpoint.get("reference_shape", [432, 432, 80]), dtype=float)
    normalized_centers = {
        side: np.asarray(center, dtype=float) / reference_shape
        for side, center in checkpoint["side_centers"].items()
    }
    model = TinyViTUNet3D(crop_size, out_channels=len(structures)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    qc_rows: list[dict] = []
    study_rows: list[dict] = []
    input_paths = sorted(args.nifti_dir.glob("*_T2.nii.gz"))
    for study_index, input_path in enumerate(input_paths, start=1):
        study_id = input_path.name.removesuffix("_T2.nii.gz")
        canonical = nib.as_closest_canonical(nib.load(str(input_path)))
        image_raw = np.asarray(canonical.dataobj, dtype=np.float32)
        spacing = tuple(float(v) for v in canonical.header.get_zooms()[:3])
        image_resampled = resample_volume(image_raw, spacing, target_spacing, order=1)
        image_resampled = normalize_intensity(image_resampled)
        new_affine = nib.affines.rescale_affine(
            canonical.affine,
            canonical.shape[:3],
            target_spacing,
            image_resampled.shape,
        )
        centers = {
            side: normalized_center * np.asarray(image_resampled.shape, dtype=float)
            for side, normalized_center in normalized_centers.items()
        }
        crops = np.stack(
            [crop_with_padding(image_resampled, centers[side], crop_size) for side in ("L", "R")],
            axis=0,
        ).astype(np.float32)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(torch.from_numpy(crops[:, None]).to(device))
            probabilities = torch.sigmoid(logits).float().cpu().numpy()

        study_dir = mask_root / f"sub{study_id}"
        study_dir.mkdir(parents=True, exist_ok=True)
        study_flags = []
        for side_index, side in enumerate(("L", "R")):
            crop_prob = probabilities[side_index]
            raw_binary = crop_prob > np.asarray(thresholds)[:, None, None, None]
            argmax_binary = raw_binary.copy()
            overlap = raw_binary.sum(axis=0) > 1
            if overlap.any():
                winning_channel = np.argmax(crop_prob, axis=0)
                for channel in range(len(structures)):
                    argmax_binary[channel, overlap] = winning_channel[overlap] == channel
            full_slices, crop_slices = crop_bounds(image_resampled.shape, centers[side], crop_size)
            for channel, structure in enumerate(structures):
                policy = policies[structure]
                source = raw_binary if policy["overlap_strategy"] == "none" else argmax_binary
                candidate = source[channel].astype(np.uint8)
                processed, component_n = retain_top_components(
                    candidate,
                    int(policy["top_k_components"]),
                )
                full_mask = np.zeros(image_resampled.shape, dtype=np.uint8)
                full_mask[full_slices] = processed[crop_slices]
                voxel_n = int(full_mask.sum())
                flags = []
                if voxel_n == 0:
                    flags.append("empty_prediction")
                elif voxel_n < 20:
                    flags.append("tiny_prediction")
                if component_n > 3:
                    flags.append("fragmented_before_lcc")
                if boundary_touch(processed):
                    flags.append("touches_crop_boundary")
                if overlap.any():
                    flags.append(
                        "channel_overlap_present_retained"
                        if policy["overlap_strategy"] == "none"
                        else "channel_overlap_resolved_argmax"
                    )
                warning_flags = [
                    flag
                    for flag in flags
                    if flag
                    not in {"channel_overlap_present_retained", "channel_overlap_resolved_argmax"}
                ]
                confidence = (
                    float(crop_prob[channel][processed > 0].mean()) if processed.any() else 0.0
                )
                output_path = study_dir / f"{study_id}{side}_{structure}.nii.gz"
                nib.save(nib.Nifti1Image(full_mask, new_affine), str(output_path))
                qc_rows.append(
                    {
                        "study_id": study_id,
                        "ear_side": side,
                        "structure": structure,
                        "threshold": thresholds[channel],
                        "predicted_voxels": voxel_n,
                        "predicted_volume_mm3": voxel_n * float(np.prod(target_spacing)),
                        "mean_foreground_probability": confidence,
                        "components_before_largest_component": component_n,
                        "touches_crop_boundary": boundary_touch(processed),
                        "overlap_voxels_across_channels": int(overlap.sum()),
                        "qc_status": "warning" if warning_flags else "pass",
                        "qc_flags": ";".join(flags),
                        "mask_path": str(output_path.resolve()),
                    }
                )
                study_flags.extend(warning_flags)
        study_rows.append(
            {
                "study_id": study_id,
                "input_nifti": str(input_path.resolve()),
                "resampled_shape": "x".join(map(str, image_resampled.shape)),
                "left_center_voxels": ",".join(f"{value:.1f}" for value in centers["L"]),
                "right_center_voxels": ",".join(f"{value:.1f}" for value in centers["R"]),
                "qc_status": "warning" if study_flags else "pass",
                "qc_flags": ";".join(sorted(set(study_flags))),
            }
        )
        print(f"INFERENCE_PROGRESS {study_index}/{len(input_paths)} {study_id}", flush=True)

    write_csv(args.output_dir / "external_mask_qc.csv", qc_rows)
    write_csv(args.output_dir / "external_study_qc.csv", study_rows)
    summary = {
        "device": str(device),
        "study_count": len(study_rows),
        "ear_count": len(study_rows) * 2,
        "mask_count": len(qc_rows),
        "mask_qc_status_counts": dict(Counter(row["qc_status"] for row in qc_rows)),
        "study_qc_status_counts": dict(Counter(row["qc_status"] for row in study_rows)),
        "structures": list(structures),
        "thresholds": dict(zip(structures, thresholds, strict=True)),
        "postprocess_policy": policies,
        "external_validation_boundary": (
            "Frozen inference plus technical/anatomical QC only; no external manual masks were "
            "available for Dice, IoU, or surface-distance validation."
        ),
    }
    (args.output_dir / "external_inference_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("INFERENCE_COMPLETE", json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
