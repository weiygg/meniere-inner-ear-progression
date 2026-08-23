from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import nibabel as nib
import SimpleITK as sitk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert extracted external DICOM series to canonical NIfTI.")
    parser.add_argument("--dicom-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metadata_value(path: str, tag: str) -> str:
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.LoadPrivateTagsOff()
    reader.ReadImageInformation()
    return reader.GetMetaData(tag).strip() if reader.HasMetaDataKey(tag) else ""


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for study_dir in sorted(item for item in args.dicom_root.iterdir() if item.is_dir()):
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(study_dir)) or []
        if len(series_ids) != 1:
            rows.append(
                {
                    "study_id": study_dir.name,
                    "series_uid": "",
                    "dicom_slices": 0,
                    "output_nifti": "",
                    "status": "failed",
                    "qc_flags": f"series_count_{len(series_ids)}",
                }
            )
            continue
        series_uid = series_ids[0]
        files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(study_dir), series_uid)
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)
        try:
            image = reader.Execute()
            temp_path = args.output_dir / f".{study_dir.name}.sitk.nii.gz"
            output_path = args.output_dir / f"{study_dir.name}_T2.nii.gz"
            sitk.WriteImage(image, str(temp_path), True)
            canonical = nib.as_closest_canonical(nib.load(str(temp_path)))
            nib.save(canonical, str(output_path))
            temp_path.unlink(missing_ok=True)
            qc_flags = []
            if len(files) < 60:
                qc_flags.append("low_slice_count")
            if max(image.GetSpacing()) > 0.75:
                qc_flags.append("coarse_spacing")
            rows.append(
                {
                    "study_id": study_dir.name,
                    "series_uid": series_uid,
                    "patient_id_dicom": metadata_value(files[0], "0010|0020"),
                    "study_date": metadata_value(files[0], "0008|0020"),
                    "series_description": metadata_value(files[0], "0008|103e"),
                    "dicom_slices": len(files),
                    "size_x": image.GetSize()[0],
                    "size_y": image.GetSize()[1],
                    "size_z": image.GetSize()[2],
                    "spacing_x_mm": image.GetSpacing()[0],
                    "spacing_y_mm": image.GetSpacing()[1],
                    "spacing_z_mm": image.GetSpacing()[2],
                    "output_nifti": str(output_path.resolve()),
                    "status": "warning" if qc_flags else "pass",
                    "qc_flags": ";".join(qc_flags),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "study_id": study_dir.name,
                    "series_uid": series_uid,
                    "dicom_slices": len(files),
                    "output_nifti": "",
                    "status": "failed",
                    "qc_flags": f"{type(exc).__name__}: {exc}",
                }
            )
        print(f"CONVERT_PROGRESS {len(rows)}/{len(list(args.dicom_root.iterdir()))} {study_dir.name}", flush=True)
    write_csv(args.output_dir.parent / "z2_series_manifest.csv", rows)
    summary = {
        "study_directories": len(rows),
        "converted": sum(bool(row.get("output_nifti")) for row in rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "low_slice_studies": [row["study_id"] for row in rows if "low_slice_count" in row["qc_flags"]],
        "validation_boundary": (
            "No external manual semicircular-canal masks were provided. These volumes support "
            "frozen inference and technical/anatomical QC only."
        ),
    }
    (args.output_dir.parent / "z2_series_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("CONVERSION_COMPLETE", json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
