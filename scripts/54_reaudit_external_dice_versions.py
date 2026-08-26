from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
FORMAL_DIR = ROOT / "results_md_progression/final/external_manual_validation_20260817"
FORMAL_METRICS = FORMAL_DIR / "external_segmentation_metrics.csv"
OUTPUT_DIR = ROOT / "results_md_progression/final/external_dice_reaudit_20260826"
STRUCTURES = ("SSC", "HSC", "PSC")

CANDIDATE_ROOTS = {
    "legacy_v1": ROOT / "results_md_progression/final/semicircular_canal_vit_20260731/external_inference/predicted_masks",
    "archived_v2": ROOT / "results_md_progression/final/semicircular_canal_vit_20260731/external_inference_v2_ensemble/predicted_masks",
    "all_t2_center2": ROOT / "results_md_progression/final/all_t2_inner_ear_vit_20260801/external_center2/predicted_masks",
    "all_t2_center3": ROOT / "results_md_progression/final/all_t2_inner_ear_vit_20260801/external_center3/predicted_masks",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prediction_and_reference(prediction_path: Path, reference_path: Path) -> tuple[np.ndarray, np.ndarray]:
    prediction_image = nib.load(str(prediction_path))
    reference_image = nib.load(str(reference_path))
    if reference_image.shape != prediction_image.shape or not np.allclose(
        reference_image.affine, prediction_image.affine, rtol=1e-5, atol=1e-4
    ):
        reference_image = resample_from_to(
            reference_image,
            (prediction_image.shape, prediction_image.affine),
            order=0,
        )
    prediction = np.asarray(prediction_image.dataobj) > 0
    reference = np.asarray(reference_image.dataobj) > 0
    return prediction, reference


def load_prediction_reusing_reference(
    prediction_path: Path,
    reference_path: Path,
    formal_prediction_image: nib.spatialimages.SpatialImage,
    formal_reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prediction_image = nib.load(str(prediction_path))
    if prediction_image.shape == formal_prediction_image.shape and np.allclose(
        prediction_image.affine,
        formal_prediction_image.affine,
        rtol=1e-5,
        atol=1e-4,
    ):
        return np.asarray(prediction_image.dataobj) > 0, formal_reference
    return load_prediction_and_reference(prediction_path, reference_path)


def dice(prediction: np.ndarray, reference: np.ndarray) -> float:
    intersection = int(np.logical_and(prediction, reference).sum())
    denominator = int(prediction.sum()) + int(reference.sum())
    return (2 * intersection + 1e-5) / (denominator + 1e-5)


def candidate_path(row: pd.Series, candidate: str) -> Path | None:
    if candidate == "formal_frozen":
        return Path(str(row["prediction_path"]))
    if candidate == "all_t2":
        root = CANDIDATE_ROOTS["all_t2_center2" if row["cohort"] == "Center 2" else "all_t2_center3"]
    else:
        root = CANDIDATE_ROOTS[candidate]
    return root / f"sub{row['patient_id']}" / f"{row['patient_id']}{row['ear_side']}_{row['structure']}.nii.gz"


def best_integer_shift_dice(
    prediction: np.ndarray,
    reference: np.ndarray,
    radius: int = 2,
) -> tuple[float, tuple[int, int, int]]:
    ref_coords = np.argwhere(reference)
    pred_count = int(prediction.sum())
    if pred_count == 0 or len(ref_coords) == 0:
        return dice(prediction, reference), (0, 0, 0)
    ref_count = len(ref_coords)
    best_score = -1.0
    best_shift = (0, 0, 0)
    shape = np.asarray(prediction.shape)
    for shift in itertools.product(range(-radius, radius + 1), repeat=3):
        shifted = ref_coords + np.asarray(shift)
        valid = np.all((shifted >= 0) & (shifted < shape), axis=1)
        valid_shifted = shifted[valid]
        intersection = int(prediction[tuple(valid_shifted.T)].sum())
        score = (2 * intersection + 1e-5) / (pred_count + ref_count + 1e-5)
        if score > best_score:
            best_score = float(score)
            best_shift = tuple(int(value) for value in shift)
    return best_score, best_shift


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in frame["candidate"].drop_duplicates():
        current = frame.loc[frame["candidate"] == candidate]
        for cohort in (*current["cohort"].drop_duplicates().tolist(), "Pooled external"):
            subset = current if cohort == "Pooled external" else current.loc[current["cohort"] == cohort]
            if subset.empty:
                continue
            for structure in (*STRUCTURES, "Macro"):
                block = subset if structure == "Macro" else subset.loc[subset["structure"] == structure]
                patient_values = block.groupby(["cohort", "patient_id"], sort=True)["dice"].mean()
                rows.append(
                    {
                        "candidate": candidate,
                        "cohort": cohort,
                        "structure": structure,
                        "patient_n": int(block[["cohort", "patient_id"]].drop_duplicates().shape[0]),
                        "mask_n": int(len(block)),
                        "mean_dice": float(patient_values.mean()),
                        "median_patient_dice": float(patient_values.median()),
                    }
                )
    return pd.DataFrame(rows)


def structure_assignment_audit(formal: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (cohort, patient_id, ear_side), group in formal.groupby(["cohort", "patient_id", "ear_side"], sort=True):
        by_structure = group.set_index("structure")
        predictions: dict[str, np.ndarray] = {}
        references: dict[str, np.ndarray] = {}
        for structure in STRUCTURES:
            prediction_path = Path(str(by_structure.loc[structure, "prediction_path"]))
            reference_path = Path(str(by_structure.loc[structure, "manual_mask_path"]))
            predictions[structure], references[structure] = load_prediction_and_reference(
                prediction_path, reference_path
            )
        matrix = np.zeros((3, 3), dtype=float)
        for ref_index, ref_structure in enumerate(STRUCTURES):
            for pred_index, pred_structure in enumerate(STRUCTURES):
                matrix[ref_index, pred_index] = dice(
                    predictions[pred_structure], references[ref_structure]
                )
        ref_indices, pred_indices = linear_sum_assignment(-matrix)
        assignment = {STRUCTURES[ref]: STRUCTURES[pred] for ref, pred in zip(ref_indices, pred_indices, strict=True)}
        rows.append(
            {
                "cohort": cohort,
                "patient_id": patient_id,
                "ear_side": ear_side,
                "identity_mean_dice": float(np.trace(matrix) / 3),
                "best_permutation_mean_dice": float(matrix[ref_indices, pred_indices].mean()),
                "best_assignment": ";".join(f"{key}->{value}" for key, value in assignment.items()),
                "identity_is_optimal": assignment == {name: name for name in STRUCTURES},
            }
        )
    return pd.DataFrame(rows)


def left_right_audit(formal: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (cohort, patient_id), group in formal.groupby(["cohort", "patient_id"], sort=True):
        lookup = group.set_index(["ear_side", "structure"])
        if not all((side, structure) in lookup.index for side in ("L", "R") for structure in STRUCTURES):
            continue
        predictions: dict[tuple[str, str], np.ndarray] = {}
        references: dict[tuple[str, str], np.ndarray] = {}
        for side in ("L", "R"):
            for structure in STRUCTURES:
                prediction_path = Path(str(lookup.loc[(side, structure), "prediction_path"]))
                reference_path = Path(str(lookup.loc[(side, structure), "manual_mask_path"]))
                predictions[(side, structure)], references[(side, structure)] = load_prediction_and_reference(
                    prediction_path, reference_path
                )
        correct_scores: list[float] = []
        swapped_scores: list[float] = []
        for structure in STRUCTURES:
            for side, other in (("L", "R"), ("R", "L")):
                reference = references[(side, structure)]
                correct_scores.append(dice(predictions[(side, structure)], reference))
                swapped_scores.append(dice(predictions[(other, structure)], reference))
        rows.append(
            {
                "cohort": cohort,
                "patient_id": patient_id,
                "correct_side_mean_dice": float(np.mean(correct_scores)),
                "swapped_side_mean_dice": float(np.mean(swapped_scores)),
                "swap_improves": float(np.mean(swapped_scores)) > float(np.mean(correct_scores)),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    formal = pd.read_csv(FORMAL_METRICS, dtype={"patient_id": str})
    formal["patient_id"] = formal["patient_id"].str.zfill(3)

    result_rows: list[dict[str, object]] = []
    shift_rows: list[dict[str, object]] = []
    candidates = ("formal_frozen", "archived_v2", "legacy_v1", "all_t2")
    for row_number, (_, row) in enumerate(formal.iterrows(), start=1):
        reference_path = Path(str(row["manual_mask_path"]))
        formal_path = Path(str(row["prediction_path"]))
        formal_prediction_image = nib.load(str(formal_path))
        formal_prediction, formal_reference = load_prediction_and_reference(formal_path, reference_path)
        formal_hash = sha256(formal_path)
        for candidate in candidates:
            prediction_path = candidate_path(row, candidate)
            if prediction_path is None or not prediction_path.exists():
                result_rows.append(
                    {
                        "candidate": candidate,
                        "cohort": row["cohort"],
                        "patient_id": row["patient_id"],
                        "ear_side": row["ear_side"],
                        "structure": row["structure"],
                        "dice": np.nan,
                        "prediction_exists": False,
                        "prediction_path": str(prediction_path) if prediction_path else "",
                        "same_file_as_formal_prediction": False,
                    }
                )
                continue
            if candidate == "formal_frozen":
                prediction, reference = formal_prediction, formal_reference
            else:
                prediction, reference = load_prediction_reusing_reference(
                    prediction_path,
                    reference_path,
                    formal_prediction_image,
                    formal_reference,
                )
            score = dice(prediction, reference)
            same_file = prediction_path.resolve() == formal_path.resolve() or sha256(prediction_path) == formal_hash
            result_rows.append(
                {
                    "candidate": candidate,
                    "cohort": row["cohort"],
                    "patient_id": row["patient_id"],
                    "ear_side": row["ear_side"],
                    "structure": row["structure"],
                    "dice": score,
                    "prediction_exists": True,
                    "prediction_path": str(prediction_path),
                    "same_file_as_formal_prediction": same_file,
                }
            )
            if candidate == "formal_frozen":
                best_shift_score, best_shift = best_integer_shift_dice(prediction, reference)
                shift_rows.append(
                    {
                        "cohort": row["cohort"],
                        "patient_id": row["patient_id"],
                        "ear_side": row["ear_side"],
                        "structure": row["structure"],
                        "formal_dice": score,
                        "best_shift_dice_diagnostic_only": best_shift_score,
                        "gain": best_shift_score - score,
                        "best_shift_voxels": ",".join(str(value) for value in best_shift),
                        "zero_shift_is_optimal": best_shift == (0, 0, 0),
                    }
                )
        if row_number % 25 == 0 or row_number == len(formal):
            print(f"candidate comparison: {row_number}/{len(formal)} masks", flush=True)

    results = pd.DataFrame(result_rows)
    if not results.loc[results["candidate"] == "formal_frozen", "prediction_exists"].all():
        raise RuntimeError("Formal frozen prediction is missing for at least one mask.")
    recomputed = results.loc[results["candidate"] == "formal_frozen"].sort_values(
        ["cohort", "patient_id", "ear_side", "structure"]
    )
    recorded = formal.sort_values(["cohort", "patient_id", "ear_side", "structure"])
    maximum_reproduction_difference = float(np.max(np.abs(recomputed["dice"].to_numpy() - recorded["dice"].to_numpy())))

    summary = summarize(results.dropna(subset=["dice"]))
    print("candidate comparison complete; starting structure assignment audit", flush=True)
    assignments = structure_assignment_audit(formal)
    print("structure assignment audit complete; starting left/right audit", flush=True)
    sides = left_right_audit(formal)
    print("left/right audit complete; writing outputs", flush=True)
    shifts = pd.DataFrame(shift_rows)

    results.to_csv(OUTPUT_DIR / "version_comparison_per_mask.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "version_comparison_summary.csv", index=False, encoding="utf-8-sig")
    assignments.to_csv(OUTPUT_DIR / "structure_assignment_audit.csv", index=False, encoding="utf-8-sig")
    sides.to_csv(OUTPUT_DIR / "left_right_assignment_audit.csv", index=False, encoding="utf-8-sig")
    shifts.to_csv(OUTPUT_DIR / "integer_shift_diagnostic.csv", index=False, encoding="utf-8-sig")

    pooled_macro = summary.loc[(summary["cohort"] == "Pooled external") & (summary["structure"] == "Macro")].copy()
    pooled_values = dict(zip(pooled_macro["candidate"], pooled_macro["mean_dice"], strict=True))
    availability = {
        candidate: {
            "available_masks": int(
                results.loc[
                    (results["candidate"] == candidate) & results["prediction_exists"],
                    "prediction_exists",
                ].sum()
            ),
            "expected_masks": int(len(formal)),
        }
        for candidate in candidates
    }
    audit_summary = {
        "formal_metric_maximum_absolute_reproduction_difference": maximum_reproduction_difference,
        "pooled_external_macro_dice_by_candidate": pooled_values,
        "candidate_reaches_0p78": {key: bool(value >= 0.78) for key, value in pooled_values.items()},
        "candidate_availability": availability,
        "all_t2_interpretation": (
            "The all-T2 external model contains Cochlear, TV, and Vestibular masks only; "
            "it has no SSC/HSC/PSC predictions and is not applicable to this Dice comparison."
        ),
        "structure_assignment_identity_optimal_ears": int(assignments["identity_is_optimal"].sum()),
        "structure_assignment_total_ears": int(len(assignments)),
        "mean_identity_assignment_dice": float(assignments["identity_mean_dice"].mean()),
        "mean_best_structure_permutation_dice_diagnostic": float(assignments["best_permutation_mean_dice"].mean()),
        "left_right_swap_improves_patients": int(sides["swap_improves"].sum()),
        "left_right_total_patients": int(len(sides)),
        "mean_correct_side_dice": float(sides["correct_side_mean_dice"].mean()),
        "mean_swapped_side_dice": float(sides["swapped_side_mean_dice"].mean()),
        "zero_integer_shift_optimal_masks": int(shifts["zero_shift_is_optimal"].sum()),
        "integer_shift_total_masks": int(len(shifts)),
        "formal_mean_dice": float(shifts["formal_dice"].mean()),
        "mean_best_integer_shift_dice_diagnostic_only": float(shifts["best_shift_dice_diagnostic_only"].mean()),
        "mean_integer_shift_gain_diagnostic_only": float(shifts["gain"].mean()),
        "interpretation_boundary": (
            "Version, permutation, side-swap, and integer-shift comparisons are error audits. "
            "The external set must not be used to select a new model, threshold, morphology rule, "
            "or registration shift while retaining an independent-validation claim."
        ),
    }
    (OUTPUT_DIR / "reaudit_summary.json").write_text(
        json.dumps(audit_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    candidate_lines = "\n".join(
        f"- `{name}`: pooled external macro Dice {value:.3f}; reaches 0.78: {'yes' if value >= 0.78 else 'no'}"
        for name, value in pooled_values.items()
    )
    report = f"""# External Dice re-audit

Date: 2026-08-26

## Result

The formal 300-mask Dice table was independently reproduced with a maximum absolute difference of {maximum_reproduction_difference:.3g}.

{candidate_lines}

- `all_t2`: not applicable; this model predicts Cochlear/TV/Vestibular rather than SSC/HSC/PSC.

## Mapping and geometry error checks

- Correct SSC/HSC/PSC identity assignment was already optimal in {int(assignments['identity_is_optimal'].sum())}/{len(assignments)} ears. The mean identity Dice was {assignments['identity_mean_dice'].mean():.3f}; allowing a post-hoc best structure permutation changed it to {assignments['best_permutation_mean_dice'].mean():.3f}.
- A left/right swap improved {int(sides['swap_improves'].sum())}/{len(sides)} patients. Mean Dice was {sides['correct_side_mean_dice'].mean():.3f} with the recorded sides versus {sides['swapped_side_mean_dice'].mean():.3f} after swapping.
- Zero integer shift was optimal for {int(shifts['zero_shift_is_optimal'].sum())}/{len(shifts)} masks. Searching ±2 voxels post hoc changed mean Dice from {shifts['formal_dice'].mean():.3f} to {shifts['best_shift_dice_diagnostic_only'].mean():.3f}; this is diagnostic only and cannot be reported as independent external performance.

## Interpretation boundary

The high 0.81–0.82 Dice artifacts in the repository are internal Lishui validation/test or union-localizer results, not external manual-reference Dice. No audited historical external prediction version reaches 0.78 on the same 50-patient reference set. Any new optimization informed by these external labels must be called model adaptation/development and requires a separate untouched test cohort for a new independent external-validation claim.
"""
    (OUTPUT_DIR / "EXTERNAL_DICE_REAUDIT.md").write_text(report, encoding="utf-8")
    print(json.dumps(audit_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
