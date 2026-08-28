from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inner_ear_vit_seg_experiment import crop_with_padding, normalize_intensity, resample_volume


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def selected_ids(path: Path) -> set[str]:
    return {item.name.zfill(3) for item in path.iterdir() if item.is_dir()}


def parse_center(text: str) -> np.ndarray:
    values = np.asarray([float(value) for value in text.split(",")], dtype=np.float32)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f"Invalid predicted center: {text}")
    return values


def safe_case_id(cohort: str, study_id: str, side: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]", "", study_id)
    return f"Z2{cohort}{token}{side}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare label-free nnU-Net ear crops for the two locked external strata."
    )
    parser.add_argument("--center2-study-qc", type=Path, required=True)
    parser.add_argument("--center3-study-qc", type=Path, required=True)
    parser.add_argument("--manual-center2-dir", type=Path, required=True)
    parser.add_argument("--manual-center3-dir", type=Path, required=True)
    parser.add_argument("--override-center3-study-qc", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-size", nargs=3, type=int, default=(128, 128, 48))
    parser.add_argument("--target-spacing", nargs=3, type=float, default=(0.3472222, 0.3472222, 0.5))
    args = parser.parse_args()

    crop_size = tuple(args.crop_size)
    target_spacing = tuple(args.target_spacing)
    images = args.output_dir / "imagesTs"
    images.mkdir(parents=True, exist_ok=True)

    override: dict[str, dict[str, str]] = {}
    if args.override_center3_study_qc:
        override = {item["study_id"].zfill(3): item for item in rows(args.override_center3_study_qc)}
    specs = (
        ("C2", args.center2_study_qc, selected_ids(args.manual_center2_dir)),
        ("C3", args.center3_study_qc, selected_ids(args.manual_center3_dir)),
    )
    records: list[dict[str, object]] = []
    source_hashes: dict[str, str] = {}
    for cohort, study_qc, keep in specs:
        lookup = {item["study_id"].zfill(3): item for item in rows(study_qc)}
        for study_id in sorted(keep):
            row = override.get(study_id, lookup.get(study_id)) if cohort == "C3" else lookup.get(study_id)
            if row is None:
                raise KeyError(f"Missing {cohort} study-QC row: {study_id}")
            source = Path(row["input_nifti"])
            if not source.exists():
                raise FileNotFoundError(source)
            canonical = nib.as_closest_canonical(nib.load(str(source)))
            raw = np.asarray(canonical.dataobj, dtype=np.float32)
            spacing = tuple(float(value) for value in canonical.header.get_zooms()[:3])
            resampled = normalize_intensity(resample_volume(raw, spacing, target_spacing, order=1))
            recorded_shape = tuple(int(value) for value in row["resampled_shape"].split("x"))
            if resampled.shape != recorded_shape:
                raise ValueError(
                    f"Resampled shape drift for {cohort}/{study_id}: {resampled.shape} != {recorded_shape}"
                )
            full_affine = nib.affines.rescale_affine(
                canonical.affine, canonical.shape[:3], target_spacing, resampled.shape
            )
            source_hashes[f"{cohort}:{study_id}"] = sha256(source)
            for side, field in (("L", "left_center_voxels"), ("R", "right_center_voxels")):
                center = parse_center(row[field])
                crop = crop_with_padding(resampled, center, crop_size).astype(np.float32)
                if crop.shape != crop_size:
                    raise ValueError(f"Crop shape drift for {cohort}/{study_id}/{side}: {crop.shape}")
                case_id = safe_case_id(cohort, study_id, side)
                crop_affine = np.diag([*target_spacing, 1.0]).astype(np.float64)
                nib.save(nib.Nifti1Image(crop, crop_affine), images / f"{case_id}_0000.nii.gz")
                records.append(
                    {
                        "case_id": case_id,
                        "cohort": cohort,
                        "study_id": study_id,
                        "ear_side": side,
                        "predicted_center_voxels": center.tolist(),
                        "crop_size": list(crop_size),
                        "target_spacing_mm": list(target_spacing),
                        "full_shape": list(resampled.shape),
                        "full_affine": full_affine.tolist(),
                        "source_sha256": source_hashes[f"{cohort}:{study_id}"],
                    }
                )

    metadata = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "people": len(source_hashes),
        "ears": len(records),
        "centers": {"C2": len(specs[0][2]), "C3": len(specs[1][2])},
        "roi_source": "frozen_union_localizer_predicted_centers_recorded_before_nnunet_selection",
        "manual_masks_loaded": False,
        "external_labels_loaded": False,
        "records": records,
    }
    (args.output_dir / "external_crop_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in metadata.items() if key != "records"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
