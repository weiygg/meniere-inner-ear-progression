from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to


STRUCTURES = ("SSC", "HSC", "PSC")
COLORS = {"SSC": "#e41a1c", "HSC": "#377eb8", "PSC": "#4daf4a"}


def load_on_grid(path: Path, reference: nib.spatialimages.SpatialImage, order: int) -> np.ndarray:
    image = nib.load(str(path))
    if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, rtol=1e-5, atol=1e-4):
        image = resample_from_to(image, (reference.shape, reference.affine), order=order)
    return np.asarray(image.dataobj)


def normalize(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    low, high = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)
    return np.clip((image - low) / max(high - low, 1e-6), 0, 1)


def roi_bounds(mask: np.ndarray, margin: int = 18) -> tuple[slice, slice]:
    coordinates = np.argwhere(mask)
    if len(coordinates) == 0:
        return slice(0, mask.shape[0]), slice(0, mask.shape[1])
    low = np.maximum(coordinates.min(axis=0) - margin, 0)
    high = np.minimum(coordinates.max(axis=0) + margin + 1, mask.shape)
    return slice(int(low[0]), int(high[0])), slice(int(low[1]), int(high[1]))


def montage(block: pd.DataFrame, output: Path) -> None:
    reference = nib.load(str(Path(block.iloc[0]["manual_t2_path"])))
    image = normalize(np.asarray(reference.dataobj, dtype=np.float32))
    truths = {}
    predictions = {}
    for structure in STRUCTURES:
        row = block.loc[block["structure"] == structure].iloc[0]
        truths[structure] = load_on_grid(Path(row["manual_mask_path"]), reference, 0) > 0
        predictions[structure] = load_on_grid(Path(row["prediction_path"]), reference, 0) > 0
    union = np.logical_or.reduce([*truths.values(), *predictions.values()])
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis_index, (axis, name) in enumerate(zip(axes, ("sagittal", "coronal", "axial"), strict=True)):
        counts = union.sum(axis=tuple(value for value in range(3) if value != axis_index))
        index = int(np.argmax(counts)) if counts.max() > 0 else image.shape[axis_index] // 2
        union_slice = np.rot90(np.take(union, index, axis=axis_index))
        bounds = roi_bounds(union_slice)
        base = np.rot90(np.take(image, index, axis=axis_index))[bounds]
        axis.imshow(base, cmap="gray", vmin=0, vmax=1)
        for structure in STRUCTURES:
            truth = np.rot90(np.take(truths[structure], index, axis=axis_index))[bounds]
            prediction = np.rot90(np.take(predictions[structure], index, axis=axis_index))[bounds]
            if truth.any():
                axis.contour(truth, levels=[0.5], colors=[COLORS[structure]], linewidths=1.4)
            if prediction.any():
                axis.contour(prediction, levels=[0.5], colors=[COLORS[structure]], linewidths=1.0, linestyles="dashed")
        axis.set_title(f"{name} slice {index}")
        axis.axis("off")
    figure.suptitle("GT solid; prediction dashed; SSC red, HSC blue, PSC green")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Select top/bottom external failure cases and render local overlays.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--bottom", type=int, default=10)
    args = parser.parse_args()
    frame = pd.read_csv(args.metrics, dtype={"patient_id": str})
    required = {"cohort", "patient_id", "ear_side", "structure", "dice", "manual_t2_path", "manual_mask_path", "prediction_path"}
    if missing := required - set(frame):
        raise ValueError(f"Metrics table missing columns: {sorted(missing)}")
    patient = frame.groupby(["cohort", "patient_id"], sort=True)["dice"].mean().rename("macro_dice").reset_index()
    selected = []
    for cohort, block in patient.groupby("cohort", sort=True):
        selected.append(block.nsmallest(args.bottom, "macro_dice").assign(selection="bottom"))
        selected.append(block.nlargest(args.top, "macro_dice").assign(selection="top"))
    selection = pd.concat(selected, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection.to_csv(args.output_dir / "failure_cases.csv", index=False, encoding="utf-8-sig")
    for row in selection.itertuples(index=False):
        patient_block = frame.loc[(frame["cohort"] == row.cohort) & (frame["patient_id"] == row.patient_id)]
        for ear, ear_block in patient_block.groupby("ear_side", sort=True):
            safe_cohort = str(row.cohort).replace(" ", "_")
            montage(ear_block, args.output_dir / row.selection / f"{safe_cohort}_{row.patient_id}_{ear}.png")
    print({"status": "complete", "selected_patients": len(selection), "output_dir": str(args.output_dir)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
