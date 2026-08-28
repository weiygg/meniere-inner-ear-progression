from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


CASE_PATTERN = re.compile(r"^(LSSEG\d+)([LR])$")
LABELS = {"1": "SSC", "2": "HSC", "3": "PSC"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def case_key(path_text: str) -> tuple[str, str]:
    name = Path(path_text).name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    match = CASE_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Unexpected validation case name: {name}")
    return match.group(1), match.group(2)


def summarize(summary: dict, *, repetitions: int, seed: int) -> dict[str, object]:
    by_patient: dict[str, list[dict[str, float]]] = defaultdict(list)
    sides: dict[str, set[str]] = defaultdict(set)
    for item in summary["metric_per_case"]:
        patient, side = case_key(item["prediction_file"])
        by_patient[patient].append(
            {name: float(item["metrics"][label]["Dice"]) for label, name in LABELS.items()}
        )
        sides[patient].add(side)
    incomplete = [patient for patient, values in sides.items() if values != {"L", "R"}]
    if incomplete:
        raise ValueError(f"Validation patients without both ears: {len(incomplete)}")

    patients = sorted(by_patient)
    records = [record for patient in patients for record in by_patient[patient]]
    point = {
        structure: float(np.mean([record[structure] for record in records]))
        for structure in LABELS.values()
    }
    point["macro"] = float(np.mean(list(point.values())))

    rng = np.random.default_rng(seed)
    bootstrap = {name: np.empty(repetitions, dtype=np.float64) for name in (*LABELS.values(), "macro")}
    for repetition in range(repetitions):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        sampled_records = [record for patient in sampled for record in by_patient[patient]]
        structure_values = {
            structure: float(np.mean([record[structure] for record in sampled_records]))
            for structure in LABELS.values()
        }
        for structure, value in structure_values.items():
            bootstrap[structure][repetition] = value
        bootstrap["macro"][repetition] = np.mean(list(structure_values.values()))

    estimates = {}
    for name in (*LABELS.values(), "macro"):
        estimates[name] = {
            "dice": point[name],
            "ci95_low": float(np.percentile(bootstrap[name], 2.5)),
            "ci95_high": float(np.percentile(bootstrap[name], 97.5)),
        }
    return {
        "people": len(patients),
        "ears": len(records),
        "bootstrap_unit": "patient",
        "bootstrap_repetitions": repetitions,
        "seed": seed,
        "estimates": estimates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a PHI-safe aggregate nnU-Net validation summary.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-final", type=Path, required=True)
    parser.add_argument("--checkpoint-best", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    training = json.loads(args.training_manifest.read_text(encoding="utf-8"))
    result = summarize(summary, repetitions=args.bootstrap, seed=args.seed)
    result.update(
        {
            "status": "complete",
            "experiment": training["experiment"],
            "run_mode": training["run_mode"],
            "epochs": training["num_epochs"],
            "configuration": training["configuration"],
            "trainer": training["trainer"],
            "external_labels_loaded": training["external_labels_loaded"],
            "interpretation": "internal_validation_pilot_not_external_performance",
            "summary_sha256": sha256(args.summary),
            "training_manifest_sha256": sha256(args.training_manifest),
            "checkpoint_final_sha256": sha256(args.checkpoint_final),
            "checkpoint_best_sha256": sha256(args.checkpoint_best),
            "weights_uploaded": False,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
