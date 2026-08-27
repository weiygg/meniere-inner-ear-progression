from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meniere_progression.registry import DatasetRegistry, validate_overlap_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Protocol V2 dataset and overlap registries.")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    args = parser.parse_args()
    registry = DatasetRegistry.load(args.config_dir / "dataset_registry.yaml")
    registry.validate()
    validate_overlap_file(args.config_dir / "dataset_overlap.yaml")
    print(
        json.dumps(
            {
                "status": "pass",
                "dataset_count": len(registry.datasets),
                "dataset_ids": sorted(registry.datasets),
                "LS_SEG_200__LS_CLIN_79_patient_overlap": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
