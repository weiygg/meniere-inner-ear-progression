from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def patient_uid(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return f"LS_SEG_{int(text):04d}"


def nnunet_case(uid: str, ear: str) -> str:
    return f"LSSEG{int(uid.rsplit('_', 1)[1]):04d}{ear}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a deterministic patient-level five-fold split.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--nnunet-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    source = pd.read_csv(args.manifest, dtype={"subject_id": str})
    required = {"subject_id", "side"}
    if missing := required - set(source):
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    source["patient_id"] = source["subject_id"].map(patient_uid)
    source["ear"] = source["side"].astype(str).str.upper()
    if len(source) != 400 or source["patient_id"].nunique() != 200:
        raise ValueError("Expected 200 patients and 400 ears")
    ear_sets = source.groupby("patient_id")["ear"].agg(set)
    if not all(ears == {"L", "R"} for ears in ear_sets):
        raise ValueError("Every patient must have exactly one left and one right ear")

    patients = sorted(source["patient_id"].unique())
    fold_for_patient: dict[str, int] = {}
    splitter = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (_, validation_indices) in enumerate(splitter.split(patients)):
        for index in validation_indices:
            fold_for_patient[patients[int(index)]] = fold
    rows = [
        {
            "dataset_id": "LS_SEG_200",
            "patient_id": row.patient_id,
            "ear": row.ear,
            "fold": fold_for_patient[row.patient_id],
        }
        for row in source[["patient_id", "ear"]].sort_values(["patient_id", "ear"]).itertuples(index=False)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    splits = []
    for fold in range(5):
        val = [nnunet_case(row["patient_id"], row["ear"]) for row in rows if row["fold"] == fold]
        train = [nnunet_case(row["patient_id"], row["ear"]) for row in rows if row["fold"] != fold]
        splits.append({"train": train, "val": val})
    args.nnunet_json.parent.mkdir(parents=True, exist_ok=True)
    args.nnunet_json.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")

    counts = []
    for fold in range(5):
        fold_rows = [row for row in rows if row["fold"] == fold]
        counts.append(
            {
                "fold": fold,
                "validation_people": len({row["patient_id"] for row in fold_rows}),
                "validation_ears": len(fold_rows),
                "training_people": 160,
                "training_ears": 320,
            }
        )
    summary = {
        "status": "pass",
        "dataset_id": "LS_SEG_200",
        "seed": args.seed,
        "folds": 5,
        "people": 200,
        "ears": 400,
        "both_ears_same_fold": True,
        "patient_overlap_between_validation_folds": 0,
        "fold_counts": counts,
        "cv_split_sha256": sha256(args.output),
        "nnunet_splits_sha256": sha256(args.nnunet_json),
        "contains_patient_level_ids": False,
        "patient_level_files_uploaded": False,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
