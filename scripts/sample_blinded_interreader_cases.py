from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meniere_progression.identifiers import subject_uid


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample local subject UIDs for blinded reader re-annotation.")
    parser.add_argument("input", type=Path, help="Protected local CSV with dataset_id and source_id")
    parser.add_argument("output", type=Path, help="Protected local output; do not commit")
    parser.add_argument("--per-stratum", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rng = random.Random(args.seed)
    selected: list[dict[str, str]] = []
    for dataset_id in ("Z2_SEG_EXT1", "Z2_SEG_EXT2"):
        candidates = [row for row in rows if row["dataset_id"] == dataset_id]
        if len(candidates) < args.per_stratum:
            raise ValueError(f"Insufficient candidates for {dataset_id}")
        for row in rng.sample(candidates, args.per_stratum):
            selected.append({"dataset_id": dataset_id, "subject_uid": subject_uid(dataset_id, row["source_id"])})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset_id", "subject_uid"])
        writer.writeheader()
        writer.writerows(selected)
    print(f"BLINDED_SAMPLE_WRITTEN people={len(selected)} seed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
