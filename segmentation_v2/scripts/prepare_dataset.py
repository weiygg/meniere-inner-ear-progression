from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meniere_progression.segmentation.multiclass import STRUCTURES, combine_multiclass_masks


def patient_uid(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return f"LS_SEG_{int(text):04d}"


def case_id(uid: str, ear: str) -> str:
    return f"LSSEG{int(uid.rsplit('_', 1)[1]):04d}{ear}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build all-400-ear Dataset502 for five-fold V2 CV.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cv-split", type=Path, required=True)
    parser.add_argument("--nnunet-raw", type=Path, required=True)
    parser.add_argument("--dataset-id", type=int, default=502)
    parser.add_argument("--dataset-name", default="LSSemicircularCanalsV2")
    parser.add_argument("--spacing-mm", nargs=3, type=float, default=(0.3472222, 0.3472222, 0.5))
    parser.add_argument("--overlap-policy", choices=("fail", "nearest-exclusive"), default="nearest-exclusive")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    source = pd.read_csv(args.manifest, dtype={"subject_id": str})
    split = pd.read_csv(args.cv_split, dtype={"patient_id": str})
    if len(source) != 400 or source["subject_id"].nunique() != 200:
        raise ValueError("Expected 200 patients and 400 ears in source manifest")
    if len(split) != 400 or split["patient_id"].nunique() != 200:
        raise ValueError("Expected 200 patients and 400 ears in CV split")
    source["patient_id"] = source["subject_id"].map(patient_uid)
    source["ear"] = source["side"].astype(str).str.upper()
    merged = source.merge(split[["patient_id", "ear", "fold"]], on=["patient_id", "ear"], validate="one_to_one")
    if len(merged) != 400:
        raise ValueError("Source manifest and CV split did not join one-to-one")

    dataset = args.nnunet_raw / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    images = dataset / "imagesTr"
    labels = dataset / "labelsTr"
    if dataset.exists() and any(dataset.iterdir()) and not args.resume:
        raise FileExistsError(f"Dataset directory is not empty: {dataset}; use --resume after auditing it")
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    affine = np.diag([*args.spacing_mm, 1.0]).astype(np.float64)
    overlap_voxels = 0
    overlap_ties = 0
    assignments = {name: 0 for name in STRUCTURES}
    written = 0
    for index, row in enumerate(merged.itertuples(index=False), start=1):
        case = case_id(row.patient_id, row.ear)
        image_path = images / f"{case}_0000.nii.gz"
        label_path = labels / f"{case}.nii.gz"
        with np.load(row.crop_path) as data:
            image = np.asarray(data["image"], dtype=np.float32)
            label, audit = combine_multiclass_masks(
                data["mask"], overlap_policy=args.overlap_policy, sampling_mm=tuple(args.spacing_mm)
            )
        if image.shape != (128, 128, 48) or label.shape != image.shape:
            raise ValueError(f"Unexpected crop shape for {case}: {image.shape}/{label.shape}")
        if not (args.resume and image_path.exists() and label_path.exists()):
            nib.save(nib.Nifti1Image(image, affine), image_path)
            nib.save(nib.Nifti1Image(label, affine), label_path)
            written += 1
        overlap_voxels += int(audit["overlap_voxels"])
        overlap_ties += int(audit["overlap_tie_voxels"])
        for name, value in audit["overlap_assignment_voxels"].items():
            assignments[name] += int(value)
        if index % 40 == 0 or index == len(merged):
            print(f"DATASET502_PROGRESS {index}/{len(merged)}", flush=True)

    splits = []
    for fold in range(5):
        val = [case_id(row.patient_id, row.ear) for row in merged.itertuples(index=False) if int(row.fold) == fold]
        train = [case_id(row.patient_id, row.ear) for row in merged.itertuples(index=False) if int(row.fold) != fold]
        splits.append({"train": sorted(train), "val": sorted(val)})
    dataset_json = {
        "channel_names": {"0": "T2"},
        "labels": {"background": 0, "SSC": 1, "HSC": 2, "PSC": 3},
        "numTraining": 400,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIOWithReorient",
    }
    (dataset / "dataset.json").write_text(json.dumps(dataset_json, indent=2) + "\n", encoding="utf-8")
    (dataset / "splits_final.json.template").write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    summary = {
        "status": "complete",
        "dataset": dataset.name,
        "people": 200,
        "ears": 400,
        "written_cases": written,
        "five_patient_level_folds": True,
        "fold_validation_people": 40,
        "fold_validation_ears": 80,
        "overlap_policy": args.overlap_policy,
        "overlap_voxels": overlap_voxels,
        "overlap_tie_voxels": overlap_ties,
        "overlap_assignment_voxels": assignments,
        "spacing_mm": list(args.spacing_mm),
        "patient_level_outputs_uploaded": False,
    }
    (dataset / "conversion_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
