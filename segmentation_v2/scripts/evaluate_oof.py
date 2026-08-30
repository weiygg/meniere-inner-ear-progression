from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


CASE = re.compile(r"^(LSSEG\d+)([LR])$")
LABELS = {"1": "SSC", "2": "HSC", "3": "PSC"}


def case_id(path_text: str) -> tuple[str, str]:
    name = Path(path_text).name.removesuffix(".nii.gz")
    match = CASE.fullmatch(name)
    if match is None:
        raise ValueError(f"Unexpected case name: {name}")
    return match.group(1), match.group(2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pool five disjoint nnU-Net validation folds into OOF Dice.")
    parser.add_argument("--summaries", nargs=5, type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--experiment-summary", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    records: list[dict[str, object]] = []
    for fold, path in enumerate(args.summaries):
        summary = json.loads(path.read_text(encoding="utf-8"))
        for item in summary["metric_per_case"]:
            patient, ear = case_id(item["prediction_file"])
            row: dict[str, object] = {"fold": fold, "patient": patient, "ear": ear}
            for label, structure in LABELS.items():
                row[structure] = float(item["metrics"][label]["Dice"])
            row["macro"] = float(np.mean([row[name] for name in LABELS.values()]))
            records.append(row)
    frame = pd.DataFrame(records)
    if len(frame) != 400 or frame[["patient", "ear"]].duplicated().any():
        raise ValueError("OOF set must contain 400 unique ears")
    if frame["patient"].nunique() != 200:
        raise ValueError("OOF set must contain 200 patients")
    if (frame.groupby("patient")["fold"].nunique() != 1).any():
        raise ValueError("Both ears of every patient must share one OOF fold")
    patient_blocks = {patient: block for patient, block in frame.groupby("patient", sort=True)}
    patients = sorted(patient_blocks)
    point = {metric: float(frame[metric].mean()) for metric in (*LABELS.values(), "macro")}
    rng = np.random.default_rng(args.seed)
    bootstrap = {metric: np.empty(args.bootstrap) for metric in point}
    for repetition in range(args.bootstrap):
        sampled = rng.choice(patients, len(patients), replace=True)
        sampled_frame = pd.concat([patient_blocks[patient] for patient in sampled], ignore_index=True)
        for metric in point:
            bootstrap[metric][repetition] = sampled_frame[metric].mean()
    estimates = {
        metric: {
            "estimate": point[metric],
            "ci95_low": float(np.percentile(values, 2.5)),
            "ci95_high": float(np.percentile(values, 97.5)),
        }
        for metric, values in bootstrap.items()
    }
    result = {
        "status": "complete",
        "experiment_id": args.experiment_id,
        "architecture": args.architecture,
        "people": 200,
        "ears": 400,
        "folds": 5,
        "bootstrap_unit": "patient",
        "bootstrap_repetitions": args.bootstrap,
        "seed": args.seed,
        "estimates": estimates,
        "external_labels_loaded": False,
        "patient_level_results_uploaded": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary_rows = []
    for fold in range(5):
        block = frame.loc[frame["fold"] == fold]
        summary_rows.append(
            {
                "experiment_id": args.experiment_id,
                "architecture": args.architecture,
                "multiclass": True,
                "fold": fold,
                "SSC_Dice": block["SSC"].mean(),
                "HSC_Dice": block["HSC"].mean(),
                "PSC_Dice": block["PSC"].mean(),
                "Macro_Dice": block["macro"].mean(),
            }
        )
    summary_rows.append(
        {
            "experiment_id": args.experiment_id,
            "architecture": args.architecture,
            "multiclass": True,
            "fold": "OOF",
            "SSC_Dice": point["SSC"],
            "HSC_Dice": point["HSC"],
            "PSC_Dice": point["PSC"],
            "Macro_Dice": point["macro"],
        }
    )
    args.experiment_summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(args.experiment_summary, index=False, encoding="utf-8-sig")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
