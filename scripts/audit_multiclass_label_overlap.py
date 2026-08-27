from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meniere_progression.segmentation.multiclass import STRUCTURES, overlap_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate-only audit of three binary canal labels.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    required = {"crop_path", "split"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Manifest lacks {sorted(required - set(frame.columns))}")
    totals = {
        "status": "complete",
        "ears": int(len(frame)),
        "people": int(frame["subject_id"].astype(str).nunique()),
        "any_overlap_ears": 0,
        "overlap_voxels": 0,
        "triple_overlap_voxels": 0,
        "pair_overlap_voxels": {"SSC_HSC": 0, "SSC_PSC": 0, "HSC_PSC": 0},
        "split_ears": {str(k): int(v) for k, v in frame["split"].value_counts().sort_index().items()},
        "interpretation": "multiclass_conversion_requires_explicit_overlap_policy",
    }
    for crop_path in frame["crop_path"]:
        with np.load(crop_path) as data:
            audit = overlap_audit(data["mask"])
        totals["any_overlap_ears"] += int(audit["overlap_voxels"] > 0)
        totals["overlap_voxels"] += audit["overlap_voxels"]
        totals["triple_overlap_voxels"] += audit["triple_overlap_voxels"]
        for pair, value in audit["pair_overlap_voxels"].items():
            totals["pair_overlap_voxels"][pair] += value

    totals["structures"] = list(STRUCTURES)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(totals, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(totals, ensure_ascii=False))


if __name__ == "__main__":
    main()
