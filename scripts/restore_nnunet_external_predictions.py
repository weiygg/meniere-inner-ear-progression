from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np


STRUCTURES = {1: "SSC", 2: "HSC", 3: "PSC"}


def crop_bounds(
    full_shape: tuple[int, int, int],
    center: np.ndarray,
    crop_size: tuple[int, int, int],
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    full_slices = []
    crop_slices = []
    for axis, size in enumerate(crop_size):
        raw_start = int(round(float(center[axis]) - size / 2))
        raw_end = raw_start + size
        full_start = max(0, raw_start)
        full_end = min(full_shape[axis], raw_end)
        crop_start = full_start - raw_start
        crop_end = crop_start + full_end - full_start
        full_slices.append(slice(full_start, full_end))
        crop_slices.append(slice(crop_start, crop_end))
    return tuple(full_slices), tuple(crop_slices)


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Restore nnU-Net external ear crops to full grids.")
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    mask_root = args.output_dir / "predicted_masks"
    mask_root.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, list[dict[str, object]]] = {"C2": [], "C3": []}
    for record in metadata["records"]:
        prediction_path = args.prediction_dir / f"{record['case_id']}.nii.gz"
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)
        prediction = np.asarray(nib.as_closest_canonical(nib.load(str(prediction_path))).dataobj)
        crop_size = tuple(int(value) for value in record["crop_size"])
        if prediction.shape != crop_size:
            raise ValueError(f"Prediction shape drift for {record['case_id']}: {prediction.shape}")
        full_shape = tuple(int(value) for value in record["full_shape"])
        center = np.asarray(record["predicted_center_voxels"], dtype=np.float32)
        full_slices, crop_slices = crop_bounds(full_shape, center, crop_size)
        study_dir = mask_root / f"sub{record['study_id']}"
        study_dir.mkdir(parents=True, exist_ok=True)
        for label, structure in STRUCTURES.items():
            full_mask = np.zeros(full_shape, dtype=np.uint8)
            full_mask[full_slices] = (prediction[crop_slices] == label).astype(np.uint8)
            output = study_dir / f"{record['study_id']}{record['ear_side']}_{structure}.nii.gz"
            nib.save(nib.Nifti1Image(full_mask, np.asarray(record["full_affine"])), output)
            manifests[record["cohort"]].append(
                {
                    "center": record["cohort"],
                    "study_id": record["study_id"],
                    "ear_side": record["ear_side"],
                    "structure": structure,
                    "inference_source": "internally_selected_M2_nnunet_frozen_evaluation",
                    "predicted_voxels": int(full_mask.sum()),
                    "mask_path": str(output.resolve()),
                }
            )
    for cohort, rows in manifests.items():
        write_manifest(args.output_dir / f"{cohort.lower()}_prediction_manifest.csv", rows)
    summary = {
        "status": "complete",
        "people": metadata["people"],
        "ears": metadata["ears"],
        "masks": sum(len(value) for value in manifests.values()),
        "external_labels_loaded": False,
        "prediction_grid": "restored_to_frozen_resampled_full_image_grid",
    }
    (args.output_dir / "restoration_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
