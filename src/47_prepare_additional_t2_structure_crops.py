from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inner_ear_vit_seg3_experiment import scan_dataset
from inner_ear_vit_seg_experiment import crop_with_padding, load_nifti, resample_volume, resize_to_shape


STRUCTURES = ("Cochlear", "Vestibular", "TV")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare predicted-ROI crops for additional T2 inner-ear masks.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/丽水-xjj内耳分割4/xjj内耳分割2"),
    )
    parser.add_argument(
        "--canal-manifest",
        type=Path,
        default=Path(
            "results_md_progression/final/semicircular_canal_vit_20260731/"
            "model_v2_structure_ensemble/sample_manifest.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_md_progression/intermediate/all_t2_vit_20260801/additional_training_crops"),
    )
    parser.add_argument("--crop-size", nargs=3, type=int, default=(128, 128, 48))
    parser.add_argument("--target-spacing", nargs=3, type=float, default=(0.3472222, 0.3472222, 0.5))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_mask(path: Path, reference_shape: tuple[int, int, int]) -> np.ndarray:
    data, _ = load_nifti(path)
    mask = (data > 0.5).astype(np.uint8)
    return (resize_to_shape(mask, reference_shape, order=0) > 0.5).astype(np.uint8)


def main() -> None:
    args = parse_args()
    crop_size = tuple(args.crop_size)
    target_spacing = tuple(args.target_spacing)
    canal_rows = {row["sample_id"]: row for row in read_csv(args.canal_manifest)}
    output_rows: list[dict[str, object]] = []
    audits: dict[str, object] = {}

    for structure in STRUCTURES:
        samples, audit = scan_dataset(args.data_dir, (structure,))
        audits[structure] = audit
        structure_dir = args.output_dir / "crops" / structure
        structure_dir.mkdir(parents=True, exist_ok=True)
        for index, sample in enumerate(samples, start=1):
            base = canal_rows.get(sample.sample_id)
            if base is None:
                raise RuntimeError(f"Missing canal ROI row for {sample.sample_id}")
            with np.load(base["crop_path"]) as crop:
                image = crop["image"].astype(np.float32)
            raw_image, spacing = load_nifti(sample.image_path)
            mask = load_mask(sample.mask_paths[0], raw_image.shape)
            resampled = (resample_volume(mask, spacing, target_spacing, order=0) > 0.5).astype(np.uint8)
            center = np.asarray(
                [base["predicted_center_x"], base["predicted_center_y"], base["predicted_center_z"]],
                dtype=np.float32,
            )
            cropped_mask = crop_with_padding(resampled, center, crop_size).astype(np.uint8)
            full_voxels = int(resampled.sum())
            crop_voxels = int(cropped_mask.sum())
            save_path = structure_dir / f"{sample.sample_id}.npz"
            np.savez_compressed(save_path, image=image, mask=cropped_mask[None])
            output_rows.append(
                {
                    "structure": structure,
                    "sample_id": sample.sample_id,
                    "subject_id": sample.subject_id,
                    "side": sample.side,
                    "split": base["split"],
                    "crop_path": str(save_path.resolve()),
                    "full_voxels": full_voxels,
                    "crop_voxels": crop_voxels,
                    "coverage": 0.0 if full_voxels == 0 else crop_voxels / full_voxels,
                }
            )
            if index % 25 == 0 or index == len(samples):
                print(f"CROP_PROGRESS {structure} {index}/{len(samples)}", flush=True)

    write_csv(args.output_dir / "sample_manifest.csv", output_rows)
    summary = {
        "structures": {
            structure: {
                "ears": sum(row["structure"] == structure for row in output_rows),
                "patients": len({row["subject_id"] for row in output_rows if row["structure"] == structure}),
                "split_ears": dict(
                    Counter(
                        row["split"] for row in output_rows if row["structure"] == structure
                    )
                ),
                "minimum_crop_coverage": min(
                    float(row["coverage"]) for row in output_rows if row["structure"] == structure
                ),
            }
            for structure in STRUCTURES
        },
        "missing_labels_are_not_negative": True,
        "roi_localisation": "Frozen union ViT predicted-center crops from the established canal model.",
        "scan_audit": audits,
    }
    (args.output_dir / "preparation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("ADDITIONAL_T2_CROPS_COMPLETE", json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
