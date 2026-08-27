from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meniere_progression.segmentation.multiclass import (
    STRUCTURES,
    combine_multiclass_masks,
    locked_nnunet_split,
    nnunet_case_id,
)


def subject_uid(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return f"LS_SEG_{int(text):04d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the locked M1 nnU-Net ear-crop dataset.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--nnunet-raw", type=Path, required=True)
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--dataset-name", default="LSSemicircularCanals")
    parser.add_argument("--spacing-mm", nargs=3, type=float, default=(0.3472222, 0.3472222, 0.5))
    parser.add_argument("--overlap-policy", choices=("fail", "nearest-exclusive"), required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest, dtype={"subject_id": str})
    needed = {"subject_id", "side", "split", "crop_path"}
    if not needed.issubset(frame.columns):
        raise ValueError(f"Manifest lacks {sorted(needed - set(frame.columns))}")
    rows = [
        {
            "subject_uid": subject_uid(row.subject_id),
            "ear_side": str(row.side).upper(),
            "split": str(row.split),
            "crop_path": str(row.crop_path),
        }
        for row in frame.itertuples(index=False)
    ]
    splits = locked_nnunet_split(rows)

    dataset_dir = args.nnunet_raw / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    labels_ts = dataset_dir / "labelsTs"
    for path in (images_tr, labels_tr, images_ts, labels_ts):
        path.mkdir(parents=True, exist_ok=True)

    affine = np.diag([*args.spacing_mm, 1.0]).astype(np.float64)
    aggregate = {
        "status": "complete",
        "dataset": dataset_dir.name,
        "source_ears": len(rows),
        "training_plus_validation_ears": 0,
        "internal_benchmark_ears": 0,
        "overlap_policy": args.overlap_policy,
        "overlap_voxels": 0,
        "overlap_tie_voxels": 0,
        "overlap_assignment_voxels": {name: 0 for name in STRUCTURES},
        "spacing_mm": list(args.spacing_mm),
        "orientation": "unmirrored_native_crop_axes_with_positive_diagonal_affine",
        "internal_benchmark_excluded_from_fingerprint": True,
    }
    for row in rows:
        case = nnunet_case_id(row["subject_uid"], row["ear_side"])
        with np.load(row["crop_path"]) as data:
            image = np.asarray(data["image"], dtype=np.float32)
            label, audit = combine_multiclass_masks(
                data["mask"], overlap_policy=args.overlap_policy, sampling_mm=tuple(args.spacing_mm)
            )
        if image.shape != label.shape:
            raise ValueError(f"Image/label shape mismatch for {case}: {image.shape}/{label.shape}")
        target_images, target_labels = (
            (images_ts, labels_ts) if row["split"] == "test" else (images_tr, labels_tr)
        )
        nib.save(nib.Nifti1Image(image, affine), target_images / f"{case}_0000.nii.gz")
        nib.save(nib.Nifti1Image(label, affine), target_labels / f"{case}.nii.gz")
        key = "internal_benchmark_ears" if row["split"] == "test" else "training_plus_validation_ears"
        aggregate[key] += 1
        aggregate["overlap_voxels"] += audit["overlap_voxels"]
        aggregate["overlap_tie_voxels"] += audit["overlap_tie_voxels"]
        for name, value in audit["overlap_assignment_voxels"].items():
            aggregate["overlap_assignment_voxels"][name] += value

    dataset_json = {
        "channel_names": {"0": "T2"},
        "labels": {"background": 0, "SSC": 1, "HSC": 2, "PSC": 3},
        "numTraining": aggregate["training_plus_validation_ears"],
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIOWithReorient",
    }
    (dataset_dir / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2) + "\n", encoding="utf-8"
    )
    (dataset_dir / "splits_final.json.template").write_text(
        json.dumps(splits, indent=2) + "\n", encoding="utf-8"
    )
    (dataset_dir / "conversion_summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate))


if __name__ == "__main__":
    main()
