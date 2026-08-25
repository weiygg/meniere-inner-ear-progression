from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mdp_utils import read_ear_records  # noqa: E402
from meniere_progression.hearing_ear_rules import (  # noqa: E402
    FREQUENCIES,
    RESOLUTION_MODES,
    EarAudiometry,
    build_rule_grid,
    consensus_summary,
    pairwise_agreement,
    summarise_rule,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def build_cohort_pairs(records: list[dict[str, Any]]) -> tuple[dict[str, dict[str, tuple[EarAudiometry, EarAudiometry]]], dict[str, dict[str, int]]]:
    cohort_records: dict[str, list[dict[str, Any]]] = {
        "LS_baseline_all": [
            row for row in records if row["source_site"] == "LS" and not row["is_followup"]
        ],
        "Z2_confirmed_MD_baseline": [
            row
            for row in records
            if row["source_site"] == "Z2"
            and not row["is_followup"]
            and "-NC" not in str(row["source_subject_id"]).upper()
            and "疑似" not in str(row["source_subject_id"])
        ],
    }
    all_pairs: dict[str, dict[str, tuple[EarAudiometry, EarAudiometry]]] = {}
    cohort_flow: dict[str, dict[str, int]] = {}
    for cohort, rows in cohort_records.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["patient_id"])].append(row)
        pairs: dict[str, tuple[EarAudiometry, EarAudiometry]] = {}
        incomplete_pair = 0
        incomplete_audiometry = 0
        for patient_key, patient_rows in grouped.items():
            by_side = {str(row["ear_side"]): row for row in patient_rows}
            if len(patient_rows) != 2 or set(by_side) != {"L", "R"}:
                incomplete_pair += 1
                continue
            ears: dict[str, EarAudiometry] = {}
            for side in ("L", "R"):
                values = tuple(numeric(by_side[side].get(name)) for name in FREQUENCIES)
                if any(value is None for value in values):
                    break
                ears[side] = EarAudiometry(
                    side=side,
                    thresholds_db=tuple(float(value) for value in values),  # type: ignore[arg-type]
                    stage_present=bool(by_side[side].get("affected_ear_proxy")),
                )
            if len(ears) != 2:
                incomplete_audiometry += 1
                continue
            pairs[patient_key] = (ears["L"], ears["R"])
        all_pairs[cohort] = pairs
        cohort_flow[cohort] = {
            "source_baseline_patients": len(grouped),
            "excluded_incomplete_or_duplicate_ear_pair": incomplete_pair,
            "excluded_incomplete_four_frequency_audiometry": incomplete_audiometry,
            "complete_paired_patients": len(pairs),
        }
    return all_pairs, cohort_flow


def fmt_fraction(value: object) -> str:
    return "NA" if value is None else f"{100 * float(value):.1f}%"


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Hearing-based affected-ear rule sensitivity audit",
        "",
        "Status: **exploratory sensitivity analysis; no affected-ear rule is frozen here, and no P-EBM was run.**",
        "",
        "## Scope and safeguards",
        "",
        "- Unit: one baseline patient with a complete left-right audiometric pair.",
        "- Audiometry: thresholds at 0.5, 1, 2, and 3 kHz; the four-frequency PTA is recomputed in code.",
        "- Abnormality is defined using strict `>` comparisons at 20, 25, 30, 35, or 40 dB HL.",
        "- The grid contains 30 ear-level rules and 5 bilateral-resolution modes (150 combinations per cohort).",
        "- AAO-HNS-stage-cell presence is evaluated only as an obsolete comparator, never as the affected-ear definition.",
        "- The report and JSON contain aggregate counts only; no patient key, name, record number, or row-level assignment is exported.",
        "",
        "The [WHO World Report on Hearing](https://www.who.int/publications/i/item/9789240020481) uses a 0.5/1/2/4-kHz average for population hearing-loss grading, so its 20/35-dB boundaries are not directly reconstructable or adopted here because this workbook has 3 kHz rather than 4 kHz. The AAO-HNS 1995 guideline is documented in [PubMed](https://pubmed.ncbi.nlm.nih.gov/7675476/); a later clinical paper describes its 0.5/1/2/3-kHz arithmetic mean ([PubMed](https://pubmed.ncbi.nlm.nih.gov/12925337/)). These sources motivate sensitivity anchors only; neither source resolves this study's bilateral index-ear definition.",
        "",
        "## Cohort flow",
        "",
        "| Cohort | Baseline patients | Incomplete/duplicate pair | Incomplete four-frequency pair | Analysed paired patients |",
        "|---|---:|---:|---:|---:|",
    ]
    for cohort, flow in payload["cohort_flow"].items():
        lines.append(
            f"| {cohort} | {flow['source_baseline_patients']} | "
            f"{flow['excluded_incomplete_or_duplicate_ear_pair']} | "
            f"{flow['excluded_incomplete_four_frequency_audiometry']} | "
            f"{flow['complete_paired_patients']} |"
        )

    selected_rule_ids = {
        "pta_05123_gt_20",
        "pta_05123_gt_25",
        "pta_05123_gt_30",
        "pta_05123_gt_35",
        "pta_05123_gt_40",
        "freq_gt_25_n2",
        "lowfreq_mean_gt_25",
    }
    lines += [
        "",
        "## Prespecified summary subset",
        "",
        "The complete 150-combination results per cohort are in the companion aggregate JSON. This table shows all PTA thresholds plus two interpretable cross-family comparators.",
        "",
    ]
    for cohort, cohort_payload in payload["cohorts"].items():
        lines += [
            f"### {cohort}",
            "",
            "| Rule | Bilateral handling | One abnormal | Both abnormal | Neither abnormal | Resolved index ear | Resolved % (95% Wilson CI) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in cohort_payload["rule_results"]:
            if row["rule_id"] not in selected_rule_ids:
                continue
            low = row["resolved_fraction_wilson95_low"]
            high = row["resolved_fraction_wilson95_high"]
            ci = f"{fmt_fraction(row['resolved_fraction'])} ({fmt_fraction(low)}-{fmt_fraction(high)})"
            lines.append(
                f"| {row['rule_id']} | {row['bilateral_resolution']} | "
                f"{row['unilateral_abnormal']} | {row['bilateral_abnormal']} | "
                f"{row['neither_abnormal']} | {row['resolved_index_ear']} | {ci} |"
            )

        lines += [
            "",
            "Rule-family envelopes across all thresholds/count variants:",
            "",
            "| Family | Bilateral handling | Minimum resolved | Maximum resolved |",
            "|---|---|---:|---:|",
        ]
        grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
        for row in cohort_payload["rule_results"]:
            grouped[(row["rule_family"], row["bilateral_resolution"])].append(int(row["resolved_index_ear"]))
        for (family, mode), counts in sorted(grouped.items()):
            lines.append(f"| {family} | {mode} | {min(counts)} | {max(counts)} |")

        lines += [
            "",
            "Cross-rule consensus among all 30 ear-level rules:",
            "",
            "| Bilateral handling | All resolve + agree | All resolve + disagree | Partial + agree | Partial + disagree | None resolve |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for mode, row in cohort_payload["consensus_by_mode"].items():
            lines.append(
                f"| {mode} | {row['all_rules_resolved_and_agree']} | "
                f"{row['all_rules_resolved_but_disagree']} | "
                f"{row['partial_rules_resolved_and_agree']} | "
                f"{row['partial_rules_resolved_and_disagree']} | {row['no_rule_resolved']} |"
            )

    lines += [
        "",
        "## Interpretation boundary",
        "",
        "A high resolved fraction or agreement with the legacy stage-cell proxy is not evidence that a rule is clinically correct. The final definition still requires the study team's hearing-feature specification, examination-time rule, and documented handling of bilateral disease. In particular, `worse_pta_no_minimum` is an exploratory stress test and is not eligible for automatic adoption.",
        "",
        "Pairwise agreement and Cohen kappa for all 30 rules are provided for `strict` and `pta_gap_15` in the JSON. Agreement is calculated only among patients resolved by both rules and therefore must be read together with the common resolved denominator.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate sensitivity audit of hearing-based affected-ear rules.")
    parser.add_argument(
        "--clinical-workbook",
        type=Path,
        default=PROJECT_ROOT / "data/clinical/MD患者评估20260713.xlsx",
    )
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "data/manifests/hearing_ear_rule_sensitivity.json")
    parser.add_argument("--output-report", type=Path, default=PROJECT_ROOT / "reports/01_hearing_ear_rule_sensitivity.md")
    args = parser.parse_args()

    workbook = args.clinical_workbook.resolve()
    records = read_ear_records(workbook)
    cohort_pairs, cohort_flow = build_cohort_pairs(records)
    rules = build_rule_grid()
    payload: dict[str, Any] = {
        "status": "exploratory_sensitivity_only_no_rule_frozen_no_pebm_run",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "safe_label": "protected_clinical_workbook",
            "sha256": sha256(workbook),
            "bytes": workbook.stat().st_size,
        },
        "parameters": {
            "frequencies_khz": [0.5, 1.0, 2.0, 3.0],
            "thresholds_db": [20, 25, 30, 35, 40],
            "comparison": "strictly_greater_than",
            "ear_level_rule_count": len(rules),
            "bilateral_resolution_modes": list(RESOLUTION_MODES),
            "combinations_per_cohort": len(rules) * len(RESOLUTION_MODES),
        },
        "cohort_flow": cohort_flow,
        "cohorts": {},
        "privacy": "aggregate_only_no_patient_or_ear_identifiers",
    }

    for cohort, pairs in cohort_pairs.items():
        rule_results: list[dict[str, object]] = []
        calls_by_mode: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
        for rule in rules:
            for mode in RESOLUTION_MODES:
                summary, calls = summarise_rule(pairs, rule, mode)
                rule_results.append(summary)
                calls_by_mode[mode][rule.rule_id] = calls
        payload["cohorts"][cohort] = {
            "rule_results": rule_results,
            "consensus_by_mode": {
                mode: consensus_summary(pairs, calls_by_mode[mode]) for mode in RESOLUTION_MODES
            },
            "pairwise_agreement": {
                mode: pairwise_agreement(calls_by_mode[mode]) for mode in ("strict", "pta_gap_15")
            },
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_report.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "cohort_flow": cohort_flow,
        "rules_per_cohort": len(rules) * len(RESOLUTION_MODES),
        "output_json": str(args.output_json),
        "output_report": str(args.output_report),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
