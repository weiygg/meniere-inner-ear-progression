from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


CANALS = ("SSC", "HSC", "PSC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare auditable JSON tables for the final feature workbook.")
    parser.add_argument("--morphometry-xlsx", type=Path, required=True)
    parser.add_argument("--clinical-xlsx", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--external-feature-json", type=Path, required=True)
    parser.add_argument("--external-qc-csv", type=Path, required=True)
    parser.add_argument("--external-study-qc-csv", type=Path, required=True)
    parser.add_argument("--series-manifest-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def normalize_id(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text.replace("_", "-")


def parse_vector(value: object, length: int) -> list[float | None]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return [None] * length
    if isinstance(value, (tuple, list, np.ndarray)):
        values = list(value)
    else:
        try:
            values = list(ast.literal_eval(str(value)))
        except Exception:
            return [None] * length
    return [float(values[index]) if index < len(values) else None for index in range(length)]


def clean_scalar(value):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) or np.isinf(value) else float(value)
    if pd.isna(value):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def records_clean(frame: pd.DataFrame) -> list[dict]:
    return [
        {str(key): clean_scalar(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def clinical_table(path: Path, sheet_index: int, site: str) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name=sheet_index, dtype=str)
    fill_columns = [column for column in ("ID", "age", "sex") if column in frame.columns]
    frame[fill_columns] = frame[fill_columns].ffill()
    frame["study_id"] = frame["ID"].map(normalize_id)
    frame["ear_side"] = frame["side"].astype(str).str.upper().str.strip()
    frame["site"] = site
    for column in ("age", "sex", "CochEH", "VestEH", "VA", "ES/ED", "PTA"):
        if column in frame:
            if column == "PTA":
                formula_values = frame[column].astype(str).str.startswith("=")
                frame.loc[formula_values, column] = np.nan
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    stage_column = next((column for column in frame.columns if str(column).startswith("stage")), None)
    if stage_column:
        frame["aao_hns_stage"] = pd.to_numeric(frame[stage_column], errors="coerce")
    else:
        frame["aao_hns_stage"] = np.nan
    frequency_columns = [column for column in ("0.5kHZ", "1kHZ", "2kHZ", "3kHZ") if column in frame]
    frequency_values = frame[frequency_columns].apply(pd.to_numeric, errors="coerce")
    frame["PTA_recomputed"] = frequency_values.mean(axis=1, skipna=False)
    keep = [
        "site",
        "study_id",
        "ear_side",
        "age",
        "sex",
        "CochEH",
        "VestEH",
        "VA",
        "ES/ED",
        "aao_hns_stage",
        "PTA_recomputed",
    ]
    for column in keep:
        if column not in frame:
            frame[column] = np.nan
    return frame[keep].drop_duplicates(["site", "study_id", "ear_side"], keep="first")


def flatten_ls_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[
        frame["batch"].astype(str).str.lower().eq("seg4")
        & frame["structure"].astype(str).str.upper().isin(CANALS)
    ].copy()
    frame["cohort"] = "LS_manual_annotation"
    frame["study_id"] = frame["seg_subject_id"].astype(str).str.replace("^sub", "", regex=True)
    frame["ear_side"] = frame["ear_side"].astype(str).str.upper()
    frame["structure"] = frame["structure"].astype(str).str.upper()
    axes = frame["principal_axis_lengths_mm"].map(lambda value: parse_vector(value, 3))
    centroid = frame["centroid_mm"].map(lambda value: parse_vector(value, 3))
    for index in range(3):
        frame[f"principal_axis_{index + 1}_mm"] = axes.map(lambda values: values[index])
        frame[f"centroid_{'xyz'[index]}_mm"] = centroid.map(lambda values: values[index])
    frame = frame.rename(columns={"surface_to_volume_ratio": "surface_to_volume_ratio_per_mm"})
    keep = [
        "cohort",
        "study_id",
        "ear_side",
        "structure",
        "volume_mm3",
        "surface_area_mm2",
        "surface_to_volume_ratio_per_mm",
        "sphericity",
        "compactness",
        "maximum_3d_diameter_mm",
        "principal_axis_1_mm",
        "principal_axis_2_mm",
        "principal_axis_3_mm",
        "elongation",
        "flatness",
        "centroid_x_mm",
        "centroid_y_mm",
        "centroid_z_mm",
        "relative_path",
    ]
    return frame[keep]


def flatten_ls_centerlines(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[
        frame["batch"].astype(str).str.lower().eq("seg4")
        & frame["structure"].astype(str).str.upper().isin(CANALS)
    ].copy()
    frame["cohort"] = "LS_manual_annotation"
    frame["study_id"] = frame["seg_subject_id"].astype(str).str.replace("^sub", "", regex=True)
    frame["ear_side"] = frame["ear_side"].astype(str).str.upper()
    frame["structure"] = frame["structure"].astype(str).str.upper()
    normals = frame["plane_normal"].map(lambda value: parse_vector(value, 3))
    for index in range(3):
        frame[f"plane_normal_{'xyz'[index]}"] = normals.map(lambda values: values[index])
    keep = [
        "cohort",
        "study_id",
        "ear_side",
        "structure",
        "centerline_status",
        "failure_reason",
        "centerline_length_mm",
        "mean_diameter_mm",
        "minimum_diameter_mm",
        "mean_curvature_per_mm",
        "maximum_curvature_per_mm",
        "mean_abs_torsion_per_mm",
        "plane_rms_residual_mm",
        "plane_normal_x",
        "plane_normal_y",
        "plane_normal_z",
        "skeleton_voxel_n",
        "main_path_point_n",
        "closed_loop_skeleton",
    ]
    return frame[keep]


def flatten_ls_angles(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[frame["batch"].astype(str).str.lower().eq("seg4")].copy()
    frame["cohort"] = "LS_manual_annotation"
    frame["study_id"] = frame["seg_subject_id"].astype(str).str.replace("^sub", "", regex=True)
    frame["ear_side"] = frame["ear_side"].astype(str).str.upper()
    return frame[
        [
            "cohort",
            "study_id",
            "ear_side",
            "canal_a",
            "canal_b",
            "acute_plane_angle_degrees",
        ]
    ]


def build_wide(
    feature_long: pd.DataFrame,
    centerline_long: pd.DataFrame,
    angle_long: pd.DataFrame,
) -> pd.DataFrame:
    key_columns = ["cohort", "study_id", "ear_side"]
    rows = []
    for key, group in feature_long.groupby(key_columns, dropna=False):
        row = dict(zip(key_columns, key, strict=True))
        for _, feature in group.iterrows():
            prefix = str(feature["structure"]).upper()
            for column, value in feature.items():
                if column not in {*key_columns, "structure", "relative_path", "mask_path"}:
                    row[f"{prefix}_{column}"] = value
        rows.append(row)
    wide = pd.DataFrame(rows)
    for _, centerline in centerline_long.iterrows():
        key = tuple(centerline[column] for column in key_columns)
        match = (
            (wide["cohort"] == key[0])
            & (wide["study_id"] == key[1])
            & (wide["ear_side"] == key[2])
        )
        prefix = str(centerline["structure"]).upper()
        for column, value in centerline.items():
            if column not in {*key_columns, "structure"}:
                wide.loc[match, f"{prefix}_{column}"] = value
    for _, angle in angle_long.iterrows():
        match = (
            (wide["cohort"] == angle["cohort"])
            & (wide["study_id"] == angle["study_id"])
            & (wide["ear_side"] == angle["ear_side"])
        )
        pair = "_".join(sorted([str(angle["canal_a"]).upper(), str(angle["canal_b"]).upper()]))
        wide.loc[match, f"angle_{pair}_degrees"] = angle["acute_plane_angle_degrees"]
    return wide


def main() -> None:
    args = parse_args()
    ls_features = flatten_ls_features(pd.read_excel(args.morphometry_xlsx, sheet_name="features"))
    ls_centerlines = flatten_ls_centerlines(pd.read_excel(args.morphometry_xlsx, sheet_name="canal_centerlines"))
    ls_angles = flatten_ls_angles(pd.read_excel(args.morphometry_xlsx, sheet_name="canal_plane_angles"))
    ls_asymmetry = pd.read_excel(args.morphometry_xlsx, sheet_name="bilateral_asymmetry")
    ls_asymmetry = ls_asymmetry[
        ls_asymmetry["batch"].astype(str).str.lower().eq("seg4")
        & ls_asymmetry["structure"].astype(str).str.upper().isin(CANALS)
    ].copy()
    ls_asymmetry["cohort"] = "LS_manual_annotation"
    ls_asymmetry["study_id"] = ls_asymmetry["seg_subject_id"].astype(str).str.replace("^sub", "", regex=True)
    ls_asymmetry = ls_asymmetry[
        ["cohort", "study_id", "structure", "metric", "left_value", "right_value", "relative_absolute_difference"]
    ]

    external_payload = json.loads(args.external_feature_json.read_text(encoding="utf-8"))
    z2_features = pd.DataFrame(external_payload["external_features"])
    z2_centerlines = pd.DataFrame(external_payload["external_centerlines"])
    z2_angles = pd.DataFrame(external_payload["external_plane_angles"])
    z2_asymmetry = pd.DataFrame(external_payload["external_bilateral_asymmetry"])
    feature_errors = pd.DataFrame(external_payload["feature_errors"])

    feature_long = pd.concat([ls_features, z2_features], ignore_index=True, sort=False)
    centerline_long = pd.concat([ls_centerlines, z2_centerlines], ignore_index=True, sort=False)
    angle_long = pd.concat([ls_angles, z2_angles], ignore_index=True, sort=False)
    asymmetry_long = pd.concat([ls_asymmetry, z2_asymmetry], ignore_index=True, sort=False)
    analysis_ready = build_wide(feature_long, centerline_long, angle_long)

    clinical = pd.concat(
        [
            clinical_table(args.clinical_xlsx, 0, "LS"),
            clinical_table(args.clinical_xlsx, 2, "Z2"),
        ],
        ignore_index=True,
    )
    analysis_ready["site"] = np.where(
        analysis_ready["cohort"].eq("LS_manual_annotation"), "LS", "Z2"
    )
    analysis_ready["study_id"] = analysis_ready["study_id"].map(normalize_id)
    analysis_ready = analysis_ready.merge(
        clinical,
        on=["site", "study_id", "ear_side"],
        how="left",
        validate="many_to_one",
    )
    analysis_ready["clinical_linked"] = analysis_ready["age"].notna()

    sample_manifest = pd.read_csv(args.training_dir / "sample_manifest.csv", dtype={"subject_id": str})
    sample_manifest["study_id"] = sample_manifest["subject_id"].str.zfill(3)
    sample_manifest["ear_side"] = sample_manifest["side"]
    sample_manifest["site"] = "LS"
    ls_manifest_columns = [
        "site",
        "study_id",
        "ear_side",
        "split",
        *[f"{structure}_coverage" for structure in CANALS],
    ]
    analysis_ready = analysis_ready.merge(
        sample_manifest[ls_manifest_columns],
        on=["site", "study_id", "ear_side"],
        how="left",
        validate="one_to_one",
    )

    mask_qc = pd.read_csv(args.external_qc_csv, dtype={"study_id": str})
    mask_qc["study_id"] = mask_qc["study_id"].map(normalize_id)
    mask_qc["site"] = "Z2"
    for structure in CANALS:
        subset = mask_qc[mask_qc["structure"].str.upper().eq(structure)].copy()
        subset = subset.rename(
            columns={
                "qc_status": f"{structure}_mask_qc_status",
                "qc_flags": f"{structure}_mask_qc_flags",
                "mean_foreground_probability": f"{structure}_mean_foreground_probability",
                "analysis_eligible_without_manual_review": (
                    f"{structure}_analysis_eligible_without_manual_review"
                ),
            }
        )
        analysis_ready = analysis_ready.merge(
            subset[
                [
                    "study_id",
                    "site",
                    "ear_side",
                    f"{structure}_mask_qc_status",
                    f"{structure}_mask_qc_flags",
                    f"{structure}_mean_foreground_probability",
                    f"{structure}_analysis_eligible_without_manual_review",
                ]
            ],
            on=["site", "study_id", "ear_side"],
            how="left",
            validate="one_to_one",
        )

    study_qc = pd.read_csv(args.external_study_qc_csv, dtype={"study_id": str})
    series_manifest = pd.read_csv(args.series_manifest_csv, dtype={"study_id": str})
    study_qc["study_id"] = study_qc["study_id"].map(normalize_id)
    series_manifest["study_id"] = series_manifest["study_id"].map(normalize_id)
    study_qc["site"] = "Z2"
    series_manifest["site"] = "Z2"
    analysis_ready = analysis_ready.merge(
        study_qc[
            [
                "site",
                "study_id",
                "qc_status",
                "qc_flags",
                "analysis_eligible_without_manual_review",
            ]
        ].rename(
            columns={
                "qc_status": "external_study_qc_status",
                "qc_flags": "external_study_qc_flags",
                "analysis_eligible_without_manual_review": (
                    "external_study_analysis_eligible_without_manual_review"
                ),
            }
        ),
        on=["site", "study_id"],
        how="left",
        validate="many_to_one",
    )
    analysis_ready["analysis_eligible_without_manual_review"] = analysis_ready["site"].eq("LS")
    analysis_ready = analysis_ready.merge(
        series_manifest[["site", "study_id", "dicom_slices", "status", "qc_flags"]].rename(
            columns={"status": "dicom_qc_status", "qc_flags": "dicom_qc_flags"}
        ),
        on=["site", "study_id"],
        how="left",
        validate="many_to_one",
    )

    internal_metrics = pd.read_csv(args.training_dir / "internal_test_deployment_metrics.csv")
    internal_summary = json.loads(
        (args.training_dir / "internal_test_deployment_summary.json").read_text(encoding="utf-8")
    )
    model_summary = json.loads((args.training_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    external_inference_summary = json.loads(
        (args.external_qc_csv.parent / "external_inference_summary.json").read_text(encoding="utf-8")
    )
    feature_summary = json.loads(
        (args.external_feature_json.parent / "feature_extraction_summary.json").read_text(encoding="utf-8")
    )

    readme = [
        {
            "item": "Model",
            "value": (
                "Two-stage ViT pipeline: frozen binary canal-union locator followed by "
                "three independent binary 3D TinyViT-UNets for SSC/HSC/PSC"
            ),
        },
        {"item": "Training source", "value": "Lishui seg4, 200 subjects / 400 ears"},
        {"item": "Split", "value": "Patient-level 140/30/30 subjects for train/validation/test"},
        {"item": "External imaging", "value": f"Z2, {external_inference_summary['study_count']} studies / {external_inference_summary['ear_count']} ears"},
        {"item": "Internal test macro Dice", "value": internal_summary["macro_mean_dice"]},
        {"item": "External validation boundary", "value": external_inference_summary["external_validation_boundary"]},
        {"item": "External analysis status", "value": external_inference_summary["external_analysis_status"]},
        {"item": "Feature boundary", "value": feature_summary["interpretation_boundary"]},
        {"item": "Known input issue", "value": "Z2 archive 2-2 is truncated; study 185 has 35 slices and remains flagged."},
        {
            "item": "Clinical source",
            "value": (
                "MD患者评估20260713.xlsx: sheet 1 Lishui, sheet 3 Zhejiang Second Hospital; "
                "names and medical record numbers excluded here."
            ),
        },
    ]
    payload = {
        "readme": readme,
        "analysis_ready": records_clean(analysis_ready),
        "features_long": records_clean(feature_long),
        "centerlines_long": records_clean(centerline_long),
        "plane_angles": records_clean(angle_long),
        "bilateral_asymmetry": records_clean(asymmetry_long),
        "internal_test_metrics": records_clean(internal_metrics),
        "external_mask_qc": records_clean(mask_qc),
        "external_study_qc": records_clean(study_qc),
        "dicom_manifest": records_clean(series_manifest),
        "feature_errors": records_clean(feature_errors) if not feature_errors.empty else [],
        "summaries": {
            "model": model_summary,
            "internal_test_deployment": internal_summary,
            "external_inference": external_inference_summary,
            "feature_extraction": feature_summary,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "analysis_ready_rows": len(payload["analysis_ready"]),
                "feature_rows": len(payload["features_long"]),
                "centerline_rows": len(payload["centerlines_long"]),
                "angle_rows": len(payload["plane_angles"]),
                "clinical_linked_rows": int(analysis_ready["clinical_linked"].sum()),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
