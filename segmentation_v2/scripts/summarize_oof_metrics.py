from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = (
    "dice",
    "iou",
    "precision",
    "recall",
    "surface_dice_0p5mm",
    "surface_dice_1p0mm",
    "assd_mm",
    "hd95_mm",
    "cldice",
    "volume_ratio",
)
STRUCTURES = ("SSC", "HSC", "PSC")


def main() -> int:
    parser = argparse.ArgumentParser(description="Patient-cluster bootstrap of five-fold OOF segmentation metrics.")
    parser.add_argument("--case-metrics", nargs=5, type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    frame = pd.concat([pd.read_csv(path, dtype={"patient": str}) for path in args.case_metrics], ignore_index=True)
    if len(frame) != 1200 or frame["patient"].nunique() != 200 or frame["case"].nunique() != 400:
        raise RuntimeError("OOF metrics must contain 200 patients, 400 ears and 1200 structure masks")
    if frame[["case", "structure"]].duplicated().any() or (frame.groupby("patient")["fold"].nunique() != 1).any():
        raise RuntimeError("OOF duplication or patient-fold leakage detected")
    patient_blocks = {patient: block for patient, block in frame.groupby("patient", sort=True)}
    patients = sorted(patient_blocks)
    rng = np.random.default_rng(args.seed)
    output: dict[str, dict[str, dict[str, float]]] = {}
    rows: list[dict[str, object]] = []
    for structure in (*STRUCTURES, "Macro"):
        point_block = frame if structure == "Macro" else frame.loc[frame["structure"] == structure]
        output[structure] = {}
        for metric in METRICS:
            estimate = float(point_block[metric].mean())
            samples = np.empty(args.bootstrap, dtype=np.float64)
            for repetition in range(args.bootstrap):
                sampled = rng.choice(patients, size=len(patients), replace=True)
                sampled_frame = pd.concat([patient_blocks[patient] for patient in sampled], ignore_index=True)
                if structure != "Macro":
                    sampled_frame = sampled_frame.loc[sampled_frame["structure"] == structure]
                samples[repetition] = sampled_frame[metric].mean()
            values = {
                "estimate": estimate,
                "ci95_low": float(np.percentile(samples, 2.5)),
                "ci95_high": float(np.percentile(samples, 97.5)),
            }
            output[structure][metric] = values
            rows.append({"structure": structure, "metric": metric, **values})
    result = {
        "status": "complete",
        "experiment": "E1",
        "people": 200,
        "ears": 400,
        "folds": 5,
        "bootstrap_unit": "patient",
        "bootstrap_repetitions": args.bootstrap,
        "seed": args.seed,
        "metrics": output,
        "external_data_loaded": False,
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    table = [
        "# nnU-Net E1 five-fold OOF summary",
        "",
        "All predictions are out-of-fold; bootstrap resampling is clustered by patient.",
        "",
        "| Structure | Dice (95% CI) | Surface Dice 1 mm | ASSD mm | HD95 mm | Precision | Recall | Volume ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for structure in (*STRUCTURES, "Macro"):
        block = output[structure]
        dice_value = block["dice"]
        table.append(
            f"| {structure} | {dice_value['estimate']:.4f} ({dice_value['ci95_low']:.4f}-{dice_value['ci95_high']:.4f}) | "
            f"{block['surface_dice_1p0mm']['estimate']:.4f} | {block['assd_mm']['estimate']:.3f} | "
            f"{block['hd95_mm']['estimate']:.3f} | {block['precision']['estimate']:.4f} | "
            f"{block['recall']['estimate']:.4f} | {block['volume_ratio']['estimate']:.4f} |"
        )
    table.extend(["", "External labels were not loaded or used.", ""])
    args.output_md.write_text("\n".join(table), encoding="utf-8")
    print(json.dumps({"status": "complete", "macro_dice": output["Macro"]["dice"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
