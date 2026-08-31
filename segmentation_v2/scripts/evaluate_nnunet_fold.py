from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from meniere_progression.segmentation.metrics import (  # noqa: E402
    dice,
    precision_recall,
    soft_cldice,
    surface_summary,
)


CASE = re.compile(r"^(LSSEG\d+)([LR])$")
LABELS = {1: "SSC", 2: "HSC", 3: "PSC"}


def surface_crop(prediction: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.argwhere(prediction | reference)
    if not len(coordinates):
        return prediction, reference
    lower = np.maximum(coordinates.min(axis=0) - 3, 0)
    upper = np.minimum(coordinates.max(axis=0) + 4, prediction.shape)
    slices = tuple(slice(int(lower[axis]), int(upper[axis])) for axis in range(3))
    return prediction[slices], reference[slices]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate one completed internal nnU-Net validation fold.")
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--case-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--oof-dir", type=Path, required=True)
    args = parser.parse_args()
    predictions = sorted(args.prediction_dir.glob("LSSEG*.nii.gz"))
    if len(predictions) != 80:
        raise RuntimeError(f"Fold {args.fold} must contain 80 validation predictions, found {len(predictions)}")
    rows: list[dict[str, object]] = []
    args.oof_dir.mkdir(parents=True, exist_ok=True)
    for prediction_path in predictions:
        case = prediction_path.name.removesuffix(".nii.gz")
        match = CASE.fullmatch(case)
        if match is None:
            raise ValueError(f"Unexpected case ID: {case}")
        reference_path = args.ground_truth_dir / prediction_path.name
        if not reference_path.exists():
            raise FileNotFoundError(reference_path)
        prediction_image = nib.as_closest_canonical(nib.load(str(prediction_path)))
        reference_image = nib.as_closest_canonical(nib.load(str(reference_path)))
        if prediction_image.shape != reference_image.shape or not np.allclose(
            prediction_image.affine, reference_image.affine, atol=1e-4, rtol=1e-5
        ):
            reference_image = resample_from_to(reference_image, prediction_image, order=0)
        prediction_labels = np.asarray(prediction_image.dataobj)
        reference_labels = np.asarray(reference_image.dataobj)
        spacing = tuple(float(value) for value in prediction_image.header.get_zooms()[:3])
        voxel_volume = float(np.prod(spacing))
        for label, structure in LABELS.items():
            prediction = prediction_labels == label
            reference = reference_labels == label
            precision, recall = precision_recall(prediction, reference)
            intersection = int(np.logical_and(prediction, reference).sum())
            union = int(np.logical_or(prediction, reference).sum())
            cropped_prediction, cropped_reference = surface_crop(prediction, reference)
            surface_05 = surface_summary(cropped_prediction, cropped_reference, spacing, tolerance_mm=0.5)
            surface_10 = surface_summary(cropped_prediction, cropped_reference, spacing, tolerance_mm=1.0)
            reference_volume = float(reference.sum() * voxel_volume)
            predicted_volume = float(prediction.sum() * voxel_volume)
            rows.append(
                {
                    "patient": match.group(1),
                    "ear": match.group(2),
                    "case": case,
                    "fold": args.fold,
                    "structure": structure,
                    "dice": dice(prediction, reference),
                    "iou": (intersection + 1e-5) / (union + 1e-5),
                    "precision": precision,
                    "recall": recall,
                    "surface_dice_0p5mm": surface_05["surface_dice_1mm"],
                    "surface_dice_1p0mm": surface_10["surface_dice_1mm"],
                    "assd_mm": surface_10["ASSD_mm"],
                    "hd95_mm": surface_10["HD95_mm"],
                    "cldice": soft_cldice(prediction, reference),
                    "reference_volume_mm3": reference_volume,
                    "predicted_volume_mm3": predicted_volume,
                    "volume_ratio": predicted_volume / reference_volume if reference_volume > 0 else np.nan,
                }
            )
        shutil.copy2(prediction_path, args.oof_dir / prediction_path.name)
    frame = pd.DataFrame(rows)
    if len(frame) != 240 or frame["patient"].nunique() != 40 or frame["case"].nunique() != 80:
        raise RuntimeError("Fold metrics must contain 40 patients, 80 ears and 240 masks")
    metrics = [
        "dice",
        "iou",
        "precision",
        "recall",
        "surface_dice_0p5mm",
        "surface_dice_1p0mm",
        "assd_mm",
        "hd95_mm",
        "cldice",
        "volume_ratio",
    ]
    summary_rows = []
    for structure in (*LABELS.values(), "Macro"):
        block = frame if structure == "Macro" else frame.loc[frame["structure"] == structure]
        summary_rows.append(
            {"fold": args.fold, "structure": structure, "n_masks": len(block), **{metric: float(block[metric].mean()) for metric in metrics}}
        )
    summary = pd.DataFrame(summary_rows)
    args.case_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.case_output, index=False, encoding="utf-8-sig")
    summary.to_csv(args.summary_output, index=False, encoding="utf-8-sig")
    macro_dice = float(summary.loc[summary["structure"] == "Macro", "dice"].iloc[0])
    result = {
        "status": "complete",
        "experiment": "E1",
        "fold": args.fold,
        "people": 40,
        "ears": 80,
        "masks": 240,
        "macro_dice": macro_dice,
        "selection_source": "LS_SEG_200_internal_validation_only",
        "external_data_loaded": False,
    }
    args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
