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
    bounding_box_center,
    crop_with_padding,
    load_nifti,
    normalize_intensity,
    resample_volume,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen two-stage external ViT ensemble inference.")
    parser.add_argument("--nifti-dir", type=Path, required=True)
    parser.add_argument("--union-checkpoint", type=Path, required=True)
    parser.add_argument("--ensemble-dir", type=Path, required=True)
    parser.add_argument("--centers-checkpoint", type=Path, required=True)
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
    current_center: np.ndarray,
    crop_size: tuple[int, int, int],
    full_shape: tuple[int, int, int],
) -> np.ndarray:
    refined = current_center + local_center - np.asarray(crop_size, dtype=np.float32) / 2.0
    return np.clip(refined, 0, np.asarray(full_shape, dtype=np.float32) - 1).astype(np.float32)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_root = args.output_dir / "predicted_masks"
    mask_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    structures = ("SSC", "HSC", "PSC")
    union_state = torch.load(args.union_checkpoint, map_location=device, weights_only=False)
    union_state = union_state.get("model_state", union_state)
    # The frozen union localizer was trained with 64 through-plane voxels
    # (pos_embed length 2048), whereas the three frozen structure models use
    # 48 through-plane voxels (pos_embed length 1536). Keep both deployed
    # geometries explicit so the archived checkpoints are directly runnable.
    localizer_crop_size = (128, 128, 64)
    structure_crop_size = (128, 128, 48)
    target_spacing = (0.3472222, 0.3472222, 0.5)
    localizer = TinyViTUNet3D(localizer_crop_size).to(device)
    localizer.load_state_dict(union_state)
    localizer.eval()

    postprocessing = json.loads(
        (args.ensemble_dir / "postprocessed_metrics_summary.json").read_text(encoding="utf-8")
    )
    policies = postprocessing["policies"]
    models = {}
    for structure in structures:
        checkpoint = torch.load(
            args.ensemble_dir / f"best_{structure}_model.pt",
            map_location=device,
            weights_only=False,
        )
        model = TinyViTUNet3D(structure_crop_size).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models[structure] = model

    center_checkpoint = torch.load(args.centers_checkpoint, map_location="cpu", weights_only=False)
    reference_shape = np.asarray(center_checkpoint.get("reference_shape", [432, 432, 80]), dtype=float)
    normalized_centers = {
        side: np.asarray(center, dtype=float) / reference_shape
        for side, center in center_checkpoint["side_centers"].items()
    }

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
        fixed_centers = {
            side: normalized_center * np.asarray(image_resampled.shape, dtype=float)
            for side, normalized_center in normalized_centers.items()
        }
        initial_crops = np.stack(
            [
                crop_with_padding(image_resampled, fixed_centers[side], localizer_crop_size)
                for side in ("L", "R")
            ],
            axis=0,
        ).astype(np.float32)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            localizer_probability = torch.sigmoid(
                localizer(torch.from_numpy(initial_crops[:, None]).to(device))
            ).float().cpu().numpy()[:, 0]

        centers = {}
        localizer_rows = {}
        for side_index, side in enumerate(("L", "R")):
            localizer_binary, localizer_components = retain_top_components(
                localizer_probability[side_index] > 0.5, 3
            )
            fallback = not localizer_binary.any()
            if fallback:
                center = fixed_centers[side].astype(np.float32)
            else:
                center = local_to_global_center(
                    bounding_box_center(localizer_binary),
                    fixed_centers[side],
                    localizer_crop_size,
                    image_resampled.shape,
                )
            centers[side] = center
            localizer_rows[side] = {
                "fallback": fallback,
                "initial_components": localizer_components,
                "initial_voxels": int(localizer_binary.sum()),
                "initial_boundary_touch": boundary_touch(localizer_binary),
            }

        refinement_crops = np.stack(
            [crop_with_padding(image_resampled, centers[side], localizer_crop_size) for side in ("L", "R")],
            axis=0,
        ).astype(np.float32)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            refinement_probability = torch.sigmoid(
                localizer(torch.from_numpy(refinement_crops[:, None]).to(device))
            ).float().cpu().numpy()[:, 0]
        for side_index, side in enumerate(("L", "R")):
            refinement_binary, _ = retain_top_components(
                refinement_probability[side_index] > 0.5, 3
            )
            if refinement_binary.any():
                centers[side] = refine_global_center(
                    bounding_box_center(refinement_binary),
                    centers[side],
                    localizer_crop_size,
                    image_resampled.shape,
                )
                localizer_rows[side]["refinement_success"] = True
            else:
                localizer_rows[side]["refinement_success"] = False

        final_crops = np.stack(
            [crop_with_padding(image_resampled, centers[side], structure_crop_size) for side in ("L", "R")],
            axis=0,
        ).astype(np.float32)
        image_tensor = torch.from_numpy(final_crops[:, None]).to(device)
        probabilities = []
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            for structure in structures:
                probabilities.append(torch.sigmoid(models[structure](image_tensor)).float().cpu().numpy()[:, 0])
        probabilities = np.stack(probabilities, axis=1)

        study_dir = mask_root / f"sub{study_id}"
        study_dir.mkdir(parents=True, exist_ok=True)
        study_flags = []
        for side_index, side in enumerate(("L", "R")):
            crop_prob = probabilities[side_index]
            raw_binary = np.stack(
                [
                    crop_prob[channel] > float(policies[structure]["threshold"])
                    for channel, structure in enumerate(structures)
                ],
                axis=0,
            )
            overlap = raw_binary.sum(axis=0) > 1
            full_slices, crop_slices = crop_bounds(image_resampled.shape, centers[side], structure_crop_size)
            for channel, structure in enumerate(structures):
                policy = policies[structure]
                candidate = raw_binary[channel].astype(np.uint8)
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
                    flags.append("channel_overlap_present_retained")
                if localizer_rows[side]["fallback"]:
                    flags.append("localizer_empty_fallback_fixed_center")
                if localizer_rows[side]["initial_boundary_touch"]:
                    flags.append("localizer_initial_prediction_touches_boundary")
                if not localizer_rows[side]["refinement_success"]:
                    flags.append("localizer_refinement_empty")
                warning_flags = [
                    flag
                    for flag in flags
                    if flag not in {"channel_overlap_present_retained"}
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
                        "threshold": policy["threshold"],
                        "top_k_components": policy["top_k_components"],
                        "predicted_voxels": voxel_n,
                        "predicted_volume_mm3": voxel_n * float(np.prod(target_spacing)),
                        "mean_foreground_probability": confidence,
                        "components_before_largest_component": component_n,
                        "touches_crop_boundary": boundary_touch(processed),
                        "overlap_voxels_across_channels": int(overlap.sum()),
                        "localizer_fallback": localizer_rows[side]["fallback"],
                        "localizer_initial_voxels": localizer_rows[side]["initial_voxels"],
                        "localizer_refinement_success": localizer_rows[side]["refinement_success"],
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
        "thresholds": {structure: policies[structure]["threshold"] for structure in structures},
        "postprocess_policy": policies,
        "roi_localisation": "Frozen union ViT predicted-center with one refinement pass.",
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
