from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import SimpleITK as sitk


GROUP_TO_CENTER = {
    "浙二1-1": "center2",
    "浙二1-2": "center2",
    "浙二2-1": "center3",
    "浙二2-2": "center3",
    "浙二2例新": "center3",
}

META_TAGS = {
    "patient_id_dicom": "0010|0020",
    "study_date": "0008|0020",
    "institution_name": "0008|0080",
    "manufacturer": "0008|0070",
    "station_name": "0008|1010",
    "manufacturer_model_name": "0008|1090",
    "device_serial_number": "0018|1000",
    "magnetic_field_strength_t": "0018|0087",
    "protocol_name": "0018|1030",
    "series_description": "0008|103e",
    "study_instance_uid": "0020|000d",
    "series_instance_uid": "0020|000e",
}


def read_dicom_metadata(path: Path) -> dict[str, str]:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    values: dict[str, str] = {}
    for name, tag in META_TAGS.items():
        value = reader.GetMetaData(tag).strip() if reader.HasMetaDataKey(tag) else ""
        value = value.encode("utf-8", errors="replace").decode("utf-8")
        values[name] = value
    return values


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["center", "source_group", "study_id"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit center2/center3 DICOM cohorts without modifying source data.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_md_progression/final/study_design_corrected_20260801/audit"),
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for group_name, center in GROUP_TO_CENTER.items():
        group_dir = args.data_root / group_name
        if not group_dir.is_dir():
            errors.append({"source_group": group_name, "study_id": "", "error": "missing_group_directory"})
            continue
        for study_dir in sorted((p for p in group_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
            dicom_files = sorted(study_dir.glob("*.dcm"))
            if not dicom_files:
                errors.append({"source_group": group_name, "study_id": study_dir.name, "error": "no_dicom_files"})
                continue
            try:
                metadata = read_dicom_metadata(dicom_files[0])
            except Exception as exc:  # keep the inventory auditable even if one file is corrupt
                metadata = {name: "" for name in META_TAGS}
                errors.append({"source_group": group_name, "study_id": study_dir.name, "error": repr(exc)})
            rows.append(
                {
                    "center": center,
                    "source_group": group_name,
                    "study_id": study_dir.name,
                    "dicom_slices": len(dicom_files),
                    "complete_72_slice_series": len(dicom_files) == 72,
                    **metadata,
                }
            )

    write_csv(args.output_dir / "external_center_inventory.csv", rows)
    write_csv(args.output_dir / "external_center_inventory_errors.csv", errors)

    by_center: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_center[str(row["center"])].append(row)

    summary: dict[str, object] = {
        "center_definition": {
            "center2": ["浙二1-1", "浙二1-2"],
            "center3": ["浙二2-1", "浙二2-2", "浙二2例新"],
        },
        "interpretation": (
            "The Lishui development cohort is center1. The reorganized Zhejiang Second Hospital "
            "batches prefixed 浙二1 and 浙二2 are treated as center2 and center3, respectively, "
            "pending confirmation by DICOM acquisition metadata and linkage to the clinical workbook."
        ),
        "centers": {},
        "errors": len(errors),
    }
    center_summaries: dict[str, object] = {}
    for center, center_rows in sorted(by_center.items()):
        center_summaries[center] = {
            "studies": len(center_rows),
            "ears_expected": len(center_rows) * 2,
            "unique_patient_ids_dicom": len({str(r["patient_id_dicom"]) for r in center_rows if r["patient_id_dicom"]}),
            "followup_named_studies": [
                str(r["study_id"]) for r in center_rows if "-" in str(r["study_id"]) or "_" in str(r["study_id"])
            ],
            "slice_count_distribution": dict(Counter(int(r["dicom_slices"]) for r in center_rows)),
            "acquisition_profiles": dict(
                Counter(
                    " | ".join(
                        str(r[key]) or "<blank>"
                        for key in (
                            "institution_name",
                            "manufacturer",
                            "manufacturer_model_name",
                            "station_name",
                            "device_serial_number",
                            "magnetic_field_strength_t",
                            "series_description",
                        )
                    )
                    for r in center_rows
                )
            ),
        }
    summary["centers"] = center_summaries
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "external_center_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
