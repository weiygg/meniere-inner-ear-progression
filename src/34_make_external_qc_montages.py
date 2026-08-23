from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


COLORS = {
    "SSC": np.asarray([1.0, 0.15, 0.15]),
    "HSC": np.asarray([0.15, 1.0, 0.25]),
    "PSC": np.asarray([0.15, 0.45, 1.0]),
}


def normalize(image: np.ndarray) -> np.ndarray:
    values = image[np.isfinite(image) & (image > 0)]
    if values.size == 0:
        return np.zeros_like(image, dtype=float)
    lo, hi = np.percentile(values, [1, 99])
    return np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)


def orient_slice(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    sliced = np.take(array, index, axis=axis)
    return np.rot90(sliced)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nifti-dir", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--studies", nargs="+", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for study_id in args.studies:
        input_path = args.nifti_dir / f"{study_id}_T2.nii.gz"
        mask_dir = args.mask_root / f"sub{study_id}"
        reference_path = mask_dir / f"{study_id}L_SSC.nii.gz"
        reference = nib.load(str(reference_path))
        source = nib.as_closest_canonical(nib.load(str(input_path)))
        image = np.asarray(
            resample_from_to(source, (reference.shape, reference.affine), order=1).dataobj,
            dtype=np.float32,
        )
        image = normalize(image)
        figure, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=160)
        for row, side in enumerate(("L", "R")):
            masks = {}
            for structure in COLORS:
                path = mask_dir / f"{study_id}{side}_{structure}.nii.gz"
                masks[structure] = np.asarray(nib.load(str(path)).dataobj) > 0
            union = np.logical_or.reduce(list(masks.values()))
            center = (
                np.round(np.argwhere(union).mean(axis=0)).astype(int)
                if union.any()
                else np.asarray(reference.shape) // 2
            )
            for column, axis in enumerate((0, 1, 2)):
                base = orient_slice(image, axis, int(center[axis]))
                rgb = np.repeat(base[..., None], 3, axis=2)
                for structure, color in COLORS.items():
                    overlay = orient_slice(masks[structure], axis, int(center[axis]))
                    rgb[overlay] = 0.35 * rgb[overlay] + 0.65 * color
                axes[row, column].imshow(rgb, origin="lower")
                axes[row, column].axis("off")
                axes[row, column].set_title(
                    f"{side} ear | {'Sagittal' if axis == 0 else 'Coronal' if axis == 1 else 'Axial'}"
                )
        figure.suptitle(
            f"Z2 {study_id} | SSC red, HSC green, PSC blue | model prediction",
            fontsize=13,
        )
        figure.tight_layout()
        figure.savefig(args.output_dir / f"{study_id}_external_qc.png", bbox_inches="tight")
        plt.close(figure)
        print(f"SAVED {study_id}", flush=True)


if __name__ == "__main__":
    main()
