from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


STRUCTURES = ("SSC", "HSC", "PSC")


def read_dice(path: Path) -> dict[tuple[str, str, str, str], float]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[tuple[str, str, str, str], float] = {}
    for row in rows:
        key = (row["cohort"], row["patient_key"], row["ear_side"], row["structure"])
        if key in result:
            raise RuntimeError(f"Duplicate comparison key: {key}")
        result[key] = float(row["dice"])
    return result


def summarize(
    deltas: dict[tuple[str, str, str, str], float],
    cohort: str | None,
    repetitions: int,
    seed: int,
) -> dict[str, object]:
    selected = {
        key: value for key, value in deltas.items() if cohort is None or key[0] == cohort
    }
    patients: dict[str, list[tuple[tuple[str, str, str, str], float]]] = defaultdict(list)
    for key, value in selected.items():
        patients[key[1]].append((key, value))
    patient_ids = sorted(patients)
    if not patient_ids:
        raise RuntimeError(f"No patients available for cohort {cohort!r}")
    rng = np.random.default_rng(seed)
    metrics: dict[str, dict[str, float]] = {}
    for structure in (*STRUCTURES, "Macro"):
        patient_values: list[float] = []
        for patient in patient_ids:
            values = [
                value
                for key, value in patients[patient]
                if structure == "Macro" or key[3] == structure
            ]
            if not values:
                raise RuntimeError(f"Missing {structure} rows for {patient}")
            patient_values.append(float(np.mean(values)))
        values_array = np.asarray(patient_values, dtype=np.float64)
        samples = rng.choice(values_array, size=(repetitions, len(values_array)), replace=True).mean(axis=1)
        metrics[structure] = {
            "candidate_minus_baseline": float(values_array.mean()),
            "ci95_low": float(np.percentile(samples, 2.5)),
            "ci95_high": float(np.percentile(samples, 97.5)),
        }
    return {
        "people": len(patient_ids),
        "paired_masks": len(selected),
        "dice_difference": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patient-clustered paired Dice comparison on an exposed external benchmark."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()

    baseline = read_dice(args.baseline)
    candidate = read_dice(args.candidate)
    if baseline.keys() != candidate.keys():
        missing_candidate = len(baseline.keys() - candidate.keys())
        missing_baseline = len(candidate.keys() - baseline.keys())
        raise RuntimeError(
            f"Paired key mismatch: missing candidate={missing_candidate}, missing baseline={missing_baseline}"
        )
    if len(baseline) != 300:
        raise RuntimeError(f"Expected 300 paired masks, found {len(baseline)}")
    deltas = {key: candidate[key] - value for key, value in baseline.items()}
    cohorts = sorted({key[0] for key in deltas})
    result = {
        "status": "complete",
        "baseline": args.baseline_label,
        "candidate": args.candidate_label,
        "comparison_status": "paired_descriptive_comparison_on_previously_exposed_same_institution_strata",
        "bootstrap_unit": "patient",
        "bootstrap_repetitions": args.bootstrap,
        "cohorts": {
            cohort: summarize(deltas, cohort, args.bootstrap, args.seed + index)
            for index, cohort in enumerate(cohorts, start=1)
        },
        "pooled": summarize(deltas, None, args.bootstrap, args.seed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["pooled"]["dice_difference"]["Macro"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
