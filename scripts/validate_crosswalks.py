from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meniere_progression.clinical.crosswalk import validate_crosswalk_fields


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an explicit, protected clinical crosswalk.")
    parser.add_argument("crosswalk", type=Path)
    args = parser.parse_args()
    with args.crosswalk.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        validate_crosswalk_fields(reader.fieldnames or [])
        count = sum(1 for _ in reader)
    print(f"CROSSWALK_SCHEMA_PASS rows={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
