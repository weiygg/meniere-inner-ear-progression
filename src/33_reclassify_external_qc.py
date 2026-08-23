from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


INFORMATIONAL = {
    "channel_overlap_present_retained",
    "channel_overlap_resolved_argmax",
    "channel_overlap_resolved",
}


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
    parser.add_argument("--inference-dir", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.inference_dir / "external_inference_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    policies = summary["postprocess_policy"]
    mask_path = args.inference_dir / "external_mask_qc.csv"
    rows = read_csv(mask_path)
    study_warnings: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        flags = [flag for flag in row["qc_flags"].split(";") if flag]
        flags = [
            (
                "channel_overlap_present_retained"
                if flag == "channel_overlap_resolved"
                and policies[row["structure"]]["overlap_strategy"] == "none"
                else "channel_overlap_resolved_argmax"
                if flag == "channel_overlap_resolved"
                else flag
            )
            for flag in flags
        ]
        warning_flags = [flag for flag in flags if flag not in INFORMATIONAL]
        row["qc_flags"] = ";".join(flags)
        row["qc_status"] = "warning" if warning_flags else "pass"
        study_warnings[row["study_id"]].update(warning_flags)
    write_csv(mask_path, rows)

    study_path = args.inference_dir / "external_study_qc.csv"
    study_rows = read_csv(study_path)
    for row in study_rows:
        warnings = sorted(study_warnings[row["study_id"]])
        row["qc_flags"] = ";".join(warnings)
        row["qc_status"] = "warning" if warnings else "pass"
    write_csv(study_path, study_rows)
    summary["mask_qc_status_counts"] = dict(Counter(row["qc_status"] for row in rows))
    summary["study_qc_status_counts"] = dict(Counter(row["qc_status"] for row in study_rows))
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
