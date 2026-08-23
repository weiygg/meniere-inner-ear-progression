from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--morphometry-xlsx", type=Path, required=True)
    parser.add_argument("--inference-dir", type=Path, required=True)
    args = parser.parse_args()
    manual = pd.read_excel(args.morphometry_xlsx, sheet_name="features")
    manual = manual[
        manual["batch"].astype(str).str.lower().eq("seg4")
        & manual["structure"].astype(str).str.upper().isin(["SSC", "HSC", "PSC"])
    ].copy()
    manual["structure"] = manual["structure"].str.upper()
    reference = {
        structure: {
            "p01": float(group["volume_mm3"].quantile(0.01)),
            "p99": float(group["volume_mm3"].quantile(0.99)),
            "median": float(group["volume_mm3"].median()),
        }
        for structure, group in manual.groupby("structure")
    }
    mask_path = args.inference_dir / "external_mask_qc.csv"
    rows = read_csv(mask_path)
    study_warnings: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        bounds = reference[row["structure"]]
        volume = float(row["predicted_volume_mm3"])
        plausible = bounds["p01"] <= volume <= bounds["p99"]
        flags = [flag for flag in row["qc_flags"].split(";") if flag]
        if not plausible:
            flags.append("volume_outside_LS_manual_p01_p99")
        row["LS_manual_volume_p01_mm3"] = bounds["p01"]
        row["LS_manual_volume_median_mm3"] = bounds["median"]
        row["LS_manual_volume_p99_mm3"] = bounds["p99"]
        row["volume_plausibility_status"] = "pass" if plausible else "warning"
        row["qc_flags"] = ";".join(sorted(set(flags)))
        warning_flags = [
            flag
            for flag in flags
            if not flag.startswith("channel_overlap_")
        ]
        row["qc_status"] = "warning" if warning_flags else "pass"
        row["analysis_eligible_without_manual_review"] = "False"
        study_warnings[row["study_id"]].update(warning_flags)
    write_csv(mask_path, rows)

    study_path = args.inference_dir / "external_study_qc.csv"
    study_rows = read_csv(study_path)
    for row in study_rows:
        warnings = sorted(study_warnings[row["study_id"]])
        row["qc_flags"] = ";".join(warnings)
        row["qc_status"] = "warning" if warnings else "pass"
        row["analysis_eligible_without_manual_review"] = "False"
    write_csv(study_path, study_rows)

    summary_path = args.inference_dir / "external_inference_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["manual_volume_reference_mm3"] = reference
    summary["mask_qc_status_counts"] = dict(Counter(row["qc_status"] for row in rows))
    summary["volume_plausibility_counts"] = dict(
        Counter(row["volume_plausibility_status"] for row in rows)
    )
    summary["study_qc_status_counts"] = dict(Counter(row["qc_status"] for row in study_rows))
    summary["external_analysis_status"] = (
        "Not eligible without manual review: sampled overlays show domain-shift failures, "
        "and no external reference masks are available."
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
