from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inner_ear_vit_seg3_experiment import CANAL_STRUCTS, scan_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the authoritative semicircular-canal training source.")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def archive_inventory(path: Path) -> dict[str, int]:
    result = subprocess.run(
        ["tar", "-tvf", str(path)],
        check=True,
        capture_output=True,
    )
    inventory: dict[str, int] = {}
    for raw_line in result.stdout.splitlines():
        fields = raw_line.split(maxsplit=8)
        if len(fields) < 9 or fields[0].startswith(b"d"):
            continue
        try:
            size = int(fields[4])
        except ValueError:
            continue
        stored_name = fields[8].decode("utf-8", errors="replace").replace("\\", "/")
        relative_name = stored_name.split("/", 1)[1] if "/" in stored_name else stored_name
        inventory[relative_name] = size
    return inventory


def local_inventory(path: Path) -> dict[str, int]:
    return {
        item.relative_to(path).as_posix(): item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    }


def affine_close(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.allclose(a, b, rtol=1e-5, atol=1e-4))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    archive_files = archive_inventory(args.archive)
    local_files = local_inventory(args.dataset)
    archive_names = set(archive_files)
    local_names = set(local_files)
    missing_local = sorted(archive_names - local_names)
    extra_local = sorted(local_names - archive_names)
    size_mismatches = sorted(
        name
        for name in archive_names & local_names
        if archive_files[name] != local_files[name]
    )
    archive_nifti = {name for name in archive_names if name.lower().endswith((".nii", ".nii.gz"))}
    local_nifti = {name for name in local_names if name.lower().endswith((".nii", ".nii.gz"))}
    missing_local_nifti = sorted(archive_nifti - local_nifti)
    extra_local_nifti = sorted(local_nifti - archive_nifti)
    nifti_size_mismatches = sorted(
        name
        for name in archive_nifti & local_nifti
        if archive_files[name] != local_files[name]
    )

    samples, scan_audit = scan_dataset(args.dataset, CANAL_STRUCTS)
    geometry_rows: list[dict] = []
    mismatch_rows: list[dict] = []
    for sample in samples:
        image = nib.load(str(sample.image_path))
        image_shape = tuple(int(v) for v in image.shape[:3])
        for structure, mask_path in zip(CANAL_STRUCTS, sample.mask_paths, strict=True):
            mask = nib.load(str(mask_path))
            mask_shape = tuple(int(v) for v in mask.shape[:3])
            shape_match = image_shape == mask_shape
            affine_match = affine_close(image.affine, mask.affine)
            row = {
                "sample_id": sample.sample_id,
                "subject_id": sample.subject_id,
                "side": sample.side,
                "structure": structure,
                "image_path": str(sample.image_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "image_shape": "x".join(map(str, image_shape)),
                "mask_shape": "x".join(map(str, mask_shape)),
                "shape_match": shape_match,
                "affine_match": affine_match,
                "mask_voxels": int(np.count_nonzero(np.asanyarray(mask.dataobj))),
            }
            geometry_rows.append(row)
            if not (shape_match and affine_match) or row["mask_voxels"] == 0:
                mismatch_rows.append(row)

    with (args.output_dir / "geometry_audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(geometry_rows[0]))
        writer.writeheader()
        writer.writerows(geometry_rows)

    summary = {
        "archive": str(args.archive.resolve()),
        "archive_sha256": sha256(args.archive),
        "dataset": str(args.dataset.resolve()),
        "archive_file_count": len(archive_files),
        "local_file_count": len(local_files),
        "name_sets_identical": not missing_local and not extra_local,
        "file_sizes_identical": not size_mismatches,
        "archive_nifti_count": len(archive_nifti),
        "local_nifti_count": len(local_nifti),
        "nifti_name_sets_identical": not missing_local_nifti and not extra_local_nifti,
        "nifti_file_sizes_identical": not nifti_size_mismatches,
        "missing_local_nifti_count": len(missing_local_nifti),
        "extra_local_nifti_count": len(extra_local_nifti),
        "nifti_size_mismatch_count": len(nifti_size_mismatches),
        "missing_local_count": len(missing_local),
        "extra_local_count": len(extra_local),
        "size_mismatch_count": len(size_mismatches),
        "missing_local_examples": missing_local[:20],
        "extra_local_examples": extra_local[:20],
        "size_mismatch_examples": size_mismatches[:20],
        "dataset_scan": scan_audit,
        "geometry_pair_count": len(geometry_rows),
        "geometry_or_empty_mismatch_count": len(mismatch_rows),
        "geometry_or_empty_mismatch_examples": mismatch_rows[:20],
        "training_source_accepted": (
            not missing_local_nifti
            and not extra_local_nifti
            and not nifti_size_mismatches
            and len(samples) == 400
            and not mismatch_rows
        ),
    }
    (args.output_dir / "source_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    if not summary["training_source_accepted"]:
        raise SystemExit("Training source audit failed; see source_audit.json")


if __name__ == "__main__":
    main()
