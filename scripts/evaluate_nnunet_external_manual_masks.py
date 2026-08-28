from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meniere_progression.segmentation.metrics import dice, precision_recall, surface_summary


STRUCTURES = ("SSC", "HSC", "PSC")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict[tuple[str, str, str], Path]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["study_id"].zfill(3), row["ear_side"], row["structure"]): Path(row["mask_path"])
        for row in rows
    }


def surface_crop(
    prediction: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.argwhere(prediction | reference)
    if not len(coordinates):
        return prediction, reference
    lower = np.maximum(coordinates.min(axis=0) - 2, 0)
    upper = np.minimum(coordinates.max(axis=0) + 3, prediction.shape)
    slices = tuple(slice(int(lower[axis]), int(upper[axis])) for axis in range(3))
    return prediction[slices], reference[slices]


def bootstrap_summary(
    rows: list[dict[str, object]], repetitions: int, seed: int
) -> dict[str, dict[str, dict[str, float]]]:
    by_patient: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_patient[str(row["patient_key"])].append(row)
    patients = sorted(by_patient)
    rng = np.random.default_rng(seed)
    metric_names = ("dice", "precision", "recall", "HD95_mm", "ASSD_mm", "surface_dice_1mm")
    result: dict[str, dict[str, dict[str, float]]] = {}
    for structure in (*STRUCTURES, "Macro"):
        point_rows = rows if structure == "Macro" else [row for row in rows if row["structure"] == structure]
        result[structure] = {}
        for metric in metric_names:
            point = float(np.mean([float(row[metric]) for row in point_rows]))
            samples = np.empty(repetitions, dtype=np.float64)
            for index in range(repetitions):
                selected = rng.choice(patients, size=len(patients), replace=True)
                sampled = [row for patient in selected for row in by_patient[patient]]
                if structure != "Macro":
                    sampled = [row for row in sampled if row["structure"] == structure]
                samples[index] = np.mean([float(row[metric]) for row in sampled])
            result[structure][metric] = {
                "estimate": point,
                "ci95_low": float(np.percentile(samples, 2.5)),
                "ci95_high": float(np.percentile(samples, 97.5)),
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen internally selected nnU-Net on two exposed external strata."
    )
    parser.add_argument("--manual-center2-dir", type=Path, required=True)
    parser.add_argument("--manual-center3-dir", type=Path, required=True)
    parser.add_argument("--manifest-center2", type=Path, required=True)
    parser.add_argument("--manifest-center3", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    cohort_results: dict[str, object] = {}
    specs = (
        ("Center 2", args.manual_center2_dir, args.manifest_center2, args.seed + 2),
        ("Center 3", args.manual_center3_dir, args.manifest_center3, args.seed + 3),
    )
    for cohort, manual_dir, manifest_path, seed in specs:
        predictions = read_manifest(manifest_path)
        cohort_rows: list[dict[str, object]] = []
        subjects = sorted(path for path in manual_dir.iterdir() if path.is_dir())
        for subject_dir in subjects:
            patient_id = subject_dir.name.zfill(3)
            for side in ("L", "R"):
                for structure in STRUCTURES:
                    manual_path = subject_dir / f"{patient_id}{side}_{structure}.nii.gz"
                    prediction_path = predictions.get((patient_id, side, structure))
                    if prediction_path is None or not prediction_path.exists() or not manual_path.exists():
                        raise FileNotFoundError(f"Missing pair: {cohort}/{patient_id}/{side}/{structure}")
                    prediction_img = nib.as_closest_canonical(nib.load(str(prediction_path)))
                    manual_img = nib.as_closest_canonical(nib.load(str(manual_path)))
                    manual_on_prediction = resample_from_to(manual_img, prediction_img, order=0)
                    prediction = np.asarray(prediction_img.dataobj) > 0
                    reference = np.asarray(manual_on_prediction.dataobj) > 0
                    spacing = tuple(float(value) for value in prediction_img.header.get_zooms()[:3])
                    precision, recall = precision_recall(prediction, reference)
                    surface_prediction, surface_reference = surface_crop(prediction, reference)
                    surface = surface_summary(surface_prediction, surface_reference, spacing)
                    union = int(np.logical_or(prediction, reference).sum())
                    intersection = int(np.logical_and(prediction, reference).sum())
                    cohort_rows.append(
                        {
                            "cohort": cohort,
                            "patient_key": f"{cohort}:{patient_id}",
                            "patient_id": patient_id,
                            "ear_side": side,
                            "structure": structure,
                            "dice": dice(prediction, reference),
                            "iou": (intersection + 1e-5) / (union + 1e-5),
                            "precision": precision,
                            "recall": recall,
                            **surface,
                        }
                    )
        if len(subjects) != 25 or len(cohort_rows) != 150:
            raise RuntimeError(f"Unexpected {cohort} denominator: {len(subjects)} people/{len(cohort_rows)} masks")
        all_rows.extend(cohort_rows)
        cohort_results[cohort] = {
            "people": len(subjects),
            "ears": len(subjects) * 2,
            "masks": len(cohort_rows),
            "estimates": bootstrap_summary(cohort_rows, args.bootstrap, seed),
        }

    pooled = bootstrap_summary(all_rows, args.bootstrap, args.seed + 4)
    result = {
        "status": "complete",
        "model": "M2 nnU-Net v2 scanner-robust five-epoch equal-budget pilot",
        "checkpoint_rule": "checkpoint_final_consistent_with_internal_validation_summary",
        "checkpoint_sha256": sha256(args.checkpoint),
        "selection_source": "LS_SEG_200_internal_validation_only",
        "external_labels_loaded_during_selection": False,
        "external_status": "previously_exposed_same_institution_strata_not_new_confirmatory_validation",
        "bootstrap_unit": "patient",
        "bootstrap_repetitions": args.bootstrap,
        "cohorts": cohort_results,
        "pooled": {"people": 50, "ears": 100, "masks": 300, "estimates": pooled},
    }
    with (args.output_dir / "external_metrics_patient_level_local.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (args.output_dir / "external_aggregate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "pooled_macro_dice": pooled["Macro"]["dice"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
