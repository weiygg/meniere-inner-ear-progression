from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


FOLLOWUP_RE = re.compile(r"(?:[_-](?:3m|6m))$", re.IGNORECASE)


def as_text(value: object) -> str:
    return "" if value is None else str(value).strip()


def as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["id"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def numeric_key(identifier: str) -> str:
    match = re.match(r"^(\d+)", identifier)
    return str(int(match.group(1))) if match else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the corrected cohort and endpoint specification.")
    parser.add_argument(
        "--sheet3-json",
        type=Path,
        default=Path("results_md_progression/final/study_design_corrected_20260801/audit/sheet3_deidentified.json"),
    )
    parser.add_argument(
        "--center-inventory",
        type=Path,
        default=Path("results_md_progression/final/study_design_corrected_20260801/audit/external_center_inventory.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_md_progression/final/study_design_corrected_20260801"),
    )
    args = parser.parse_args()

    payload = json.loads(args.sheet3_json.read_text(encoding="utf-8"))
    headers = payload["headers"]
    records = [dict(zip(headers, row, strict=True)) for row in payload["rows"]]
    visits: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        visits[as_text(record["ID"])].append(record)

    visit_rows: list[dict[str, object]] = []
    for visit_id, ear_rows in visits.items():
        is_followup = bool(FOLLOWUP_RE.search(visit_id))
        explicit_control = "-NC" in visit_id.upper()
        suspected = "疑似" in visit_id
        baseline = not is_followup
        confirmed_md_primary = baseline and not explicit_control and not suspected
        stages = [as_float(row.get("stage（AAO-HNS）")) for row in ear_rows]
        stages = [value for value in stages if value is not None]
        pta_values = [as_float(row.get("PTA")) for row in ear_rows]
        hydrops_zero_bilateral = all(
            (as_float(row.get("CochEH")) or 0) == 0 and (as_float(row.get("VestEH")) or 0) == 0
            for row in ear_rows
        )
        pta_normal_bilateral = all(value is not None and value <= 25 for value in pta_values)
        exclusion_reason = ""
        if is_followup:
            exclusion_reason = "followup_not_independent_for_cross_sectional_pebm"
        elif explicit_control:
            exclusion_reason = "explicit_NC_control_not_MD"
        elif suspected:
            exclusion_reason = "suspected_not_confirmed_MD"
        visit_rows.append(
            {
                "visit_id": visit_id,
                "numeric_patient_key": numeric_key(visit_id),
                "ear_rows": len(ear_rows),
                "baseline": baseline,
                "followup": is_followup,
                "explicit_NC_control": explicit_control,
                "suspected_case": suspected,
                "bilateral_cochlear_and_vestibular_hydrops_zero": hydrops_zero_bilateral,
                "bilateral_PTA_le_25_db": pta_normal_bilateral,
                "AAO_HNS_stage_nonmissing_ears": len(stages),
                "AAO_HNS_stage_value": stages[0] if len(stages) == 1 else "",
                "include_primary_patient_level_PEBM": confirmed_md_primary,
                "exclusion_reason": exclusion_reason,
            }
        )

    inventory = read_csv(args.center_inventory)
    clinical_by_numeric = {row["numeric_patient_key"]: row for row in visit_rows if row["baseline"] and row["numeric_patient_key"]}
    linkage_rows: list[dict[str, object]] = []
    for image in inventory:
        study_id = image["study_id"]
        key = numeric_key(study_id)
        clinical = clinical_by_numeric.get(key)
        imaging_followup = bool(FOLLOWUP_RE.search(study_id))
        baseline_link = clinical is not None and not imaging_followup
        primary_md_link = bool(
            baseline_link and clinical and clinical["include_primary_patient_level_PEBM"]
        )
        linkage_rows.append(
            {
                "center": image["center"],
                "source_group": image["source_group"],
                "study_id": study_id,
                "dicom_slices": image["dicom_slices"],
                "numeric_patient_key": key,
                "imaging_followup": imaging_followup,
                "linked_to_sheet3_baseline": baseline_link,
                "clinical_visit_id": clinical["visit_id"] if clinical else "",
                "include_primary_patient_level_PEBM": primary_md_link,
                "manual_reference_masks_available": False,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "audit" / "sheet3_visit_level_audit.csv", visit_rows)
    write_csv(args.output_dir / "audit" / "external_center_clinical_linkage.csv", linkage_rows)

    center_summary: dict[str, object] = {}
    for center in ("external_validation_1", "external_validation_2"):
        subset = [row for row in linkage_rows if row["center"] == center]
        center_summary[center] = {
            "imaging_studies": len(subset),
            "imaging_ears": len(subset) * 2,
            "sheet3_baseline_linked_studies": sum(bool(row["linked_to_sheet3_baseline"]) for row in subset),
            "primary_MD_linked_studies": sum(bool(row["include_primary_patient_level_PEBM"]) for row in subset),
            "manual_reference_mask_studies": 0,
        }

    summary = {
        "segmentation_development": {
            "site": "Lishui",
            "ears": 400,
            "unit_of_split": "patient",
        },
        "external_segmentation_and_downstream_cohorts": center_summary,
        "sheet3": {
            "ear_rows": len(records),
            "visits": len(visits),
            "baseline_people": sum(bool(row["baseline"]) for row in visit_rows),
            "followup_visits": sum(bool(row["followup"]) for row in visit_rows),
            "explicit_NC_controls": sum(bool(row["explicit_NC_control"]) for row in visit_rows),
            "suspected_cases": sum(bool(row["suspected_case"]) for row in visit_rows),
            "primary_confirmed_MD_baseline_patients": sum(
                bool(row["include_primary_patient_level_PEBM"]) for row in visit_rows
            ),
            "exclusion_reason_distribution": dict(
                Counter(row["exclusion_reason"] for row in visit_rows if row["exclusion_reason"])
            ),
        },
        "locked_endpoint_rules": {
            "AAO_HNS_stage": "supervised prediction target; excluded from P-EBM event inputs and cohort/affected-ear definition",
            "PEBM_unit": "one baseline row per confirmed MD patient; bilateral ear data aggregated patient-wise",
            "PEBM_exclusions": "follow-up visits, explicit -NC controls, and the 疑似 case in the primary analysis",
            "segmentation_external_Dice": "requires manual reference masks in each external cohort; currently unavailable",
        },
    }
    (args.output_dir / "cohort_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    design = f"""# 冻结的队列与终点规范（2026-08-01）

## 队列

- 开发队列：丽水 400 耳，仅用于分割模型训练、验证和内部测试，所有划分按患者完成。
- 外部验证 1：`浙二1-1` + `浙二1-2`，当前 {center_summary['external_validation_1']['imaging_studies']} 次影像/{center_summary['external_validation_1']['imaging_ears']} 耳。
- 外部验证 2：`浙二2-1` + `浙二2-2` + `浙二2例新`，当前 {center_summary['external_validation_2']['imaging_studies']} 次影像/{center_summary['external_validation_2']['imaging_ears']} 耳。
- 两个外部验证组均来自浙二，是相对丽水的两个预先分层外部影像队列，不表述为两个独立医院。

## 分割与外部验证

- T2 分割目标：Cochlear、Vestibular、SSC、HSC、PSC、TV。各标签允许重叠，缺失标注不作阴性。
- REAL 分割目标：ELS；当前浙二目录仅发现 T2 DICOM，故 ELS 不能在 center2/center3 推理或验证，除非补充对应 REAL 图像。
- 冻结模型后分别对外部验证 1、外部验证 2 推理。自动 mask 可用于后续特征提取前的人工 QC；若要报告外部 Dice/HD95，每个外部队列必须另有人工参考 mask。

## 监督预测

- 独立建立影像特征模型、临床模型和影像+临床融合模型。
- AAO-HNS stage 是预测终点之一，绝不作为预测它自身的输入。
- 所有缺失值处理、标准化、特征筛选、阈值选择均只在丽水/训练折内拟合，外部验证 1 和 2 不参与调参。

## P-EBM

- 来源：`MD患者评估20260713.xlsx` 第三张表（浙二），按患者基线分析。
- 表内共有 {len(records)} 个耳行、{len(visits)} 次双耳访视、{sum(bool(row['baseline']) for row in visit_rows)} 名基线患者和 {sum(bool(row['followup']) for row in visit_rows)} 次随访。
- 主要 P-EBM 队列为 {sum(bool(row['include_primary_patient_level_PEBM']) for row in visit_rows)} 名确认 MD 基线患者；排除 3 个明确 `-NC`、1 个“疑似”及 3 次非独立随访。
- AAO-HNS stage 不进入 P-EBM 事件、不定义患耳；stage 仅在监督模型中作为终点，或作为预先声明的次要外部临床关联指标。
- 双耳指标采用预先定义的患者级聚合，不能把两耳当作独立患者。
"""
    (args.output_dir / "COHORT_AND_ENDPOINT_LOCK.md").write_text(design, encoding="utf-8")

    obsolete = """# 旧结果的状态

以下既往输出保留用于审计，但不再作为正式结论：

- `clinical_pebm_external_validation_20260731`：丽水建模、浙二验证的旧定义，不符合当前指定的浙二患者级 P-EBM 队列。
- `clinical_pebm_z2_development_20260801`：使用“AAO-HNS stage 只填一侧”推断患耳的代理定义，已被当前患者级设计否决。

不得从上述目录复制 AUC、事件顺序或 stage 相关数字进入新报告。新 P-EBM 必须使用本目录冻结的患者级纳排与事件定义重新运行。
"""
    (args.output_dir / "OBSOLETE_OUTPUTS.md").write_text(obsolete, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
