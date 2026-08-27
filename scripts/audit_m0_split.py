from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meniere_progression.identifiers import subject_uid
from meniere_progression.segmentation.datasets import aggregate_split_counts, validate_patient_level_split


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the frozen M0 patient-level split manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = [
        {
            "subject_uid": subject_uid("LS_SEG_200", row["subject_id"]),
            "ear_side": row.get("side") or row.get("ear_side"),
            "split": row["split"],
        }
        for row in source_rows
    ]
    validate_patient_level_split(rows)
    result = {
        "status": "pass",
        "dataset_id": "LS_SEG_200",
        "manifest_sha256": digest(args.manifest),
        "people": len({row["subject_uid"] for row in rows}),
        "ears": len(rows),
        "split_counts": aggregate_split_counts(rows),
        "patient_level_split": True,
        "both_ears_same_split": True,
        "interpretation": "historical internal benchmark; repeated prior review means it is not described as untouched",
        "contains_patient_level_ids": False,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
