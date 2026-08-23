from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


COLORS = {
    "Cochlear": np.asarray([1.0, 0.2, 0.8]),
    "Vestibular": np.asarray([1.0, 0.75, 0.05]),
    "SSC": np.asarray([1.0, 0.1, 0.1]),
    "HSC": np.asarray([0.1, 0.95, 0.2]),
    "PSC": np.asarray([0.1, 0.4, 1.0]),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def normalize(image: np.ndarray) -> np.ndarray:
    values = image[np.isfinite(image) & (image > 0)]
    if values.size == 0:
        return np.zeros_like(image, dtype=float)
    low, high = np.percentile(values, [1, 99])
    return np.clip((image - low) / max(high - low, 1e-6), 0, 1)


def orient_slice(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.rot90(np.take(array, index, axis=axis))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create center-stratified six-structure QC montages.")
    parser.add_argument(
        "--nifti-dir",
        type=Path,
        default=Path("results_md_progression/intermediate/semicircular_canal_vit_20260731/z2_prepared/nifti"),
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=Path("results_md_progression/final/all_t2_inner_ear_vit_20260801"),
    )
    parser.add_argument("--per-center", type=int, default=6)
    args = parser.parse_args()

    for center in ("center2", "center3"):
        center_dir = args.external_root / f"external_{center}"
        manifest_rows = read_csv(center_dir / "all_six_structure_mask_manifest.csv")
        additional_qc = read_csv(center_dir / "additional_mask_qc.csv")
        study_flags: dict[str, int] = defaultdict(int)
        for row in additional_qc:
            study_flags[row["study_id"]] += int(row["qc_status"] == "warning")
        studies = sorted({row["study_id"] for row in manifest_rows})
        warnings = sorted((study for study in studies if study_flags[study] > 0), key=lambda study: (-study_flags[study], study))
        passes = [study for study in studies if study_flags[study] == 0]
        warning_n = min(len(warnings), max(1, args.per_center // 2))
        selected = warnings[:warning_n] + passes[: max(0, args.per_center - warning_n)]
        if len(selected) < args.per_center:
            selected += warnings[warning_n : warning_n + args.per_center - len(selected)]
        by_key = {(row["study_id"], row["ear_side"], row["structure"]): row for row in manifest_rows}
        output_dir = center_dir / "qc_montages"
        output_dir.mkdir(parents=True, exist_ok=True)

        for study_id in selected:
            input_path = args.nifti_dir / f"{study_id}_T2.nii.gz"
            reference_row = by_key[(study_id, "L", "SSC")]
            reference = nib.load(reference_row["mask_path"])
            source = nib.as_closest_canonical(nib.load(str(input_path)))
            image = np.asarray(resample_from_to(source, (reference.shape, reference.affine), order=1).dataobj, dtype=np.float32)
            image = normalize(image)
            figure, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=160)
            for row_index, side in enumerate(("L", "R")):
                masks = {
                    structure: np.asarray(nib.load(by_key[(study_id, side, structure)]["mask_path"]).dataobj) > 0
                    for structure in (*COLORS, "TV")
                }
                union = np.logical_or.reduce([masks[structure] for structure in COLORS])
                center_voxel = np.round(np.argwhere(union).mean(axis=0)).astype(int) if union.any() else np.asarray(reference.shape) // 2
                for column, axis in enumerate((0, 1, 2)):
                    base = orient_slice(image, axis, int(center_voxel[axis]))
                    rgb = np.repeat(base[..., None], 3, axis=2)
                    for structure, color in COLORS.items():
                        overlay = orient_slice(masks[structure], axis, int(center_voxel[axis]))
                        rgb[overlay] = 0.45 * rgb[overlay] + 0.55 * color
                    axes[row_index, column].imshow(rgb, origin="lower")
                    tv_slice = orient_slice(masks["TV"], axis, int(center_voxel[axis]))
                    if tv_slice.any():
                        axes[row_index, column].contour(tv_slice.astype(float), levels=[0.5], colors=["white"], linewidths=0.7)
                    axes[row_index, column].axis("off")
                    axes[row_index, column].set_title(
                        f"{side} | {'Sagittal' if axis == 0 else 'Coronal' if axis == 1 else 'Axial'}"
                    )
            figure.suptitle(
                f"{center} {study_id} | Cochlea magenta, Vestibule yellow, SSC red, HSC green, PSC blue, TV white contour\n"
                f"additional-QC warnings={study_flags[study_id]} | automatic masks, not manual reference",
                fontsize=11,
            )
            figure.tight_layout()
            figure.savefig(output_dir / f"{study_id}_six_structure_qc.png", bbox_inches="tight")
            plt.close(figure)
            print(f"QC_MONTAGE {center} {study_id}", flush=True)


if __name__ == "__main__":
    main()
