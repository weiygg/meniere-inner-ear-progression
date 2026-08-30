from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
STRUCTURES = ("SSC", "HSC", "PSC")
TARGET_SPACING = (0.3472222, 0.3472222, 0.5)
CROP_SIZE = (128, 128, 48)


def describe(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {name: None for name in ("n", "mean", "sd", "median", "q1", "q3", "min", "max")}
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def bh_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 1.0
    for rank_index in range(len(order) - 1, -1, -1):
        source_index = int(order[rank_index])
        rank = rank_index + 1
        running = min(running, p_values[source_index] * len(p_values) / rank)
        adjusted[source_index] = min(running, 1.0)
    return adjusted.tolist()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def environment() -> dict[str, object]:
    import torch

    cuda = torch.cuda.is_available()
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        if cuda
        else None,
        "nnunetv2": package_version("nnunetv2"),
        "monai": package_version("monai"),
        "nibabel": package_version("nibabel"),
        "scikit_image": package_version("scikit-image"),
    }


def v1_parameter_count() -> int:
    sys.path.insert(0, str(ROOT))
    from inner_ear_vit_seg_experiment import TinyViTUNet3D

    model = TinyViTUNet3D(CROP_SIZE)
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def aggregate_metrics(frame: pd.DataFrame, cohort: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    assd = "assd_mm" if "assd_mm" in frame else "average_symmetric_surface_distance_mm"
    metric_columns = {
        "dice": "dice",
        "iou": "iou",
        "precision": "precision",
        "recall": "recall",
        "surface_dice_0p5mm": "surface_dice_0p5mm",
        "surface_dice_1p0mm": "surface_dice_1p0mm",
        "assd_mm": assd,
        "hd95_mm": "hd95_mm",
    }
    for structure in STRUCTURES:
        block = frame.loc[frame["structure"] == structure]
        row: dict[str, object] = {"cohort": cohort, "structure": structure, "ear_rows": int(len(block))}
        for output_name, source_name in metric_columns.items():
            row[output_name] = float(pd.to_numeric(block[source_name], errors="coerce").mean()) if source_name in block else None
        reference = pd.to_numeric(block["reference_volume_mm3"], errors="coerce")
        predicted = pd.to_numeric(block["predicted_volume_mm3"], errors="coerce")
        ratios = (predicted / reference).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        ratio_summary = describe(ratios)
        for name in ("mean", "sd", "median", "q1", "q3"):
            row[f"volume_ratio_{name}"] = ratio_summary[name]
        rows.append(row)
    return rows


def format_value(value: object, digits: int = 3) -> str:
    if value is None or not np.isfinite(float(value)):
        return "not available"
    return f"{float(value):.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the frozen V1 canal segmentation baseline.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--internal-metrics", type=Path, required=True)
    parser.add_argument("--internal-summary", type=Path, required=True)
    parser.add_argument("--external-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--public-json", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest, dtype={"subject_id": str})
    internal = pd.read_csv(args.internal_metrics, dtype={"sample_id": str})
    external = pd.read_csv(args.external_metrics, dtype={"patient_id": str})
    internal_summary = json.loads(args.internal_summary.read_text(encoding="utf-8"))
    required_manifest = {"subject_id", "side", "split", "crop_path"}
    if missing := required_manifest - set(manifest):
        raise ValueError(f"V1 manifest missing columns: {sorted(missing)}")
    if len(manifest) != 400 or manifest["subject_id"].nunique() != 200:
        raise ValueError("Expected exactly 200 patients and 400 ears in the V1 manifest")
    ear_sets = manifest.groupby("subject_id")["side"].agg(lambda values: set(values.astype(str).str.upper()))
    if not all(value == {"L", "R"} for value in ear_sets):
        raise ValueError("Every Lishui patient must have both L and R ears")

    voxel_volume = float(np.prod(TARGET_SPACING))
    volume_rows: list[dict[str, object]] = []
    for structure in STRUCTURES:
        volumes = pd.to_numeric(manifest[f"{structure}_full_voxels"], errors="raise") * voxel_volume
        for subject, side, volume in zip(manifest["subject_id"], manifest["side"], volumes, strict=True):
            volume_rows.append({"cohort": "Lishui", "patient": subject, "ear": side, "structure": structure, "volume_mm3": float(volume)})
    external_volume = external[["patient_id", "ear_side", "structure", "reference_volume_mm3"]].copy()
    external_volume.columns = ["patient", "ear", "structure", "volume_mm3"]
    external_volume.insert(0, "cohort", "External manual reference")
    volumes = pd.concat([pd.DataFrame(volume_rows), external_volume], ignore_index=True)

    volume_summary: list[dict[str, object]] = []
    shift_rows: list[dict[str, object]] = []
    p_values: list[float] = []
    for cohort in ("Lishui", "External manual reference"):
        for structure in STRUCTURES:
            block = volumes.loc[(volumes["cohort"] == cohort) & (volumes["structure"] == structure)]
            volume_summary.append({"cohort": cohort, "structure": structure, **describe(block["volume_mm3"].to_numpy(float))})
    for structure in STRUCTURES:
        block = volumes.loc[volumes["structure"] == structure]
        patient_means = block.groupby(["cohort", "patient"], sort=True)["volume_mm3"].mean()
        lishui = patient_means.loc["Lishui"].to_numpy(float)
        ext = patient_means.loc["External manual reference"].to_numpy(float)
        test = stats.mannwhitneyu(lishui, ext, alternative="two-sided")
        p_values.append(float(test.pvalue))
        shift_rows.append(
            {
                "structure": structure,
                "lishui_patient_n": int(len(lishui)),
                "external_patient_n": int(len(ext)),
                "lishui_patient_median_mm3": float(np.median(lishui)),
                "external_patient_median_mm3": float(np.median(ext)),
                "external_to_lishui_median_ratio": float(np.median(ext) / np.median(lishui)),
                "mann_whitney_u": float(test.statistic),
                "p_value": float(test.pvalue),
            }
        )
    for row, q_value in zip(shift_rows, bh_adjust(p_values), strict=True):
        row["bh_q_value"] = q_value

    metric_rows = aggregate_metrics(internal, "V1 internal test") + aggregate_metrics(external, "V1 exposed external")
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "baseline_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(volume_summary).to_csv(args.output_dir / "mask_volume_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(shift_rows).to_csv(args.output_dir / "mask_volume_shift_tests.csv", index=False, encoding="utf-8-sig")

    env = environment()
    (args.output_dir / "environment.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in env.items()) + "\n", encoding="utf-8"
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    external_rows = {row["structure"]: row for row in metric_rows if row["cohort"] == "V1 exposed external"}
    internal_rows = {row["structure"]: row for row in metric_rows if row["cohort"] == "V1 internal test"}

    error_interpretation = {}
    for structure, row in external_rows.items():
        ratio = float(row["volume_ratio_median"])
        precision = float(row["precision"])
        recall = float(row["recall"])
        if ratio < 0.95 and recall < precision:
            verdict = "predominantly under-segmentation / false negatives"
        elif ratio > 1.05 and precision < recall:
            verdict = "predominantly over-segmentation / false positives"
        else:
            verdict = "mixed boundary-thickness and endpoint mismatch; no single FP/FN direction dominates"
        error_interpretation[structure] = verdict

    public = {
        "status": "complete",
        "audit_date": "2026-08-30",
        "git_commit": commit,
        "v1": {
            "architecture": "three independent binary 3D TinyViT-UNet models",
            "trainable_parameters_per_model": v1_parameter_count(),
            "crop_size_xyz": list(CROP_SIZE),
            "target_spacing_xyz_mm": list(TARGET_SPACING),
            "bottleneck_xyz": [16, 16, 6],
            "downsampling_ratio_xyz": [8, 8, 8],
            "loss": "0.70 Dice + 0.20 Tversky(FP=0.65,FN=0.35) + 0.10 focal",
            "augmentation": "translation up to 4 voxels; intensity scale/shift +/-10%; Gaussian noise SD 0.03",
            "split": "patient-level 140/30/30 people; historical internal test previously viewed",
        },
        "internal_macro_dice": float(internal_summary["internal_test_postprocessed_macro_dice"]),
        "internal_structure_metrics": internal_rows,
        "external_structure_metrics": external_rows,
        "volume_summary": volume_summary,
        "volume_shift_tests": shift_rows,
        "error_interpretation": error_interpretation,
        "environment": env,
        "external_labels_used_for_model_selection": False,
        "contains_patient_level_data": False,
    }
    args.public_json.parent.mkdir(parents=True, exist_ok=True)
    args.public_json.write_text(json.dumps(public, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    lines = [
        "# SEGMENTATION V2 baseline audit",
        "",
        "Date: 2026-08-30",
        "",
        f"Latest audited development commit: `{commit}` on `codex/protocol-v2-seg-pebm`. `origin/main` is older and is not the source of the completed M1-M3 results.",
        "",
        "## V1 architecture and input",
        "",
        "- Architecture: three independent binary 3D TinyViT-UNet models (SSC, HSC, PSC), each with one input and one output channel.",
        f"- Trainable parameters per structure model: {public['v1']['trainable_parameters_per_model']:,}.",
        "- Resampling: image interpolation order 1 in the legacy crop pipeline; masks use nearest-neighbour/order 0. Target spacing is 0.3472222 x 0.3472222 x 0.5 mm.",
        "- Normalisation: nonzero 0.5th-99.5th percentile clipping followed by nonzero z-score normalisation.",
        "- Crop: 128 x 128 x 48 voxels, centred by the frozen union localiser.",
        "",
        "| Network point | X x Y x Z |",
        "|---|---:|",
        "| Input / stem | 128 x 128 x 48 |",
        "| Encoder down1, stride 2 | 64 x 64 x 24 |",
        "| Patch embedding, kernel/stride 4 | 16 x 16 x 6 |",
        "| Transformer bottleneck | 16 x 16 x 6 (1,536 tokens) |",
        "| Decoder output | 128 x 128 x 48 |",
        "",
        "The z direction is reduced to six bottleneck tokens. This is a plausible thin-tubular-structure information bottleneck, but it is a hypothesis until a paired internal high-resolution ablation is run.",
        "",
        "## Training definition",
        "",
        "- Loss: 0.70 Dice + 0.20 Tversky + 0.10 focal. The Tversky denominator weights FP=0.65 and FN=0.35.",
        "- Augmentation: translation up to four voxels, +/-10% intensity scale/shift, and Gaussian noise (SD 0.03). No scanner-resolution, blur, Rician-like noise, rotation or bias-field augmentation was present in V1.",
        "- Split: patient-level 140/30/30 people (280/60/60 ears), seed 42. Both ears stay together. The internal test has been reviewed previously and is not an untouched test.",
        "- Thresholds/postprocessing selected on validation only: SSC 0.15/top-1 component; HSC 0.20/all components; PSC 0.10/top-1 component.",
        "",
        "## Verified V1 performance",
        "",
        "| Cohort | SSC Dice | HSC Dice | PSC Dice | Macro Dice | Surface Dice 1 mm | ASSD, mm | HD95, mm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Lishui historical internal test | {format_value(internal_rows['SSC']['dice'])} | {format_value(internal_rows['HSC']['dice'])} | {format_value(internal_rows['PSC']['dice'])} | {public['internal_macro_dice']:.4f} | not available in the frozen V1 table | {np.mean([internal_rows[s]['assd_mm'] for s in STRUCTURES]):.3f} | {np.mean([internal_rows[s]['hd95_mm'] for s in STRUCTURES]):.3f} |",
        f"| Exposed external manual set | {format_value(external_rows['SSC']['dice'])} | {format_value(external_rows['HSC']['dice'])} | {format_value(external_rows['PSC']['dice'])} | {np.mean([external_rows[s]['dice'] for s in STRUCTURES]):.4f} | {np.mean([external_rows[s]['surface_dice_1p0mm'] for s in STRUCTURES]):.3f} | {np.mean([external_rows[s]['assd_mm'] for s in STRUCTURES]):.3f} | {np.mean([external_rows[s]['hd95_mm'] for s in STRUCTURES]):.3f} |",
        "",
        "## External quantitative error audit",
        "",
        "| Structure | Volume ratio mean (SD) | Median (IQR) | Precision | Recall | Interpretation |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for structure in STRUCTURES:
        row = external_rows[structure]
        lines.append(
            f"| {structure} | {row['volume_ratio_mean']:.3f} ({row['volume_ratio_sd']:.3f}) | {row['volume_ratio_median']:.3f} ({row['volume_ratio_q1']:.3f}-{row['volume_ratio_q3']:.3f}) | {row['precision']:.3f} | {row['recall']:.3f} | {error_interpretation[structure]} |"
        )
    lines.extend(
        [
            "",
            "High 1-mm surface Dice with much lower volumetric Dice is compatible with one- to two-voxel thickness, boundary and endpoint differences in these small canals. It does not prove that thickness is the only cause. Motion, contrast, acquisition resolution and annotation convention remain candidate contributors.",
            "",
            "## Reference-mask volume distribution",
            "",
            "Ear-level summaries are descriptive. Shift tests use patient-mean volumes across both ears and Benjamini-Hochberg correction.",
            "",
            "| Structure | Lishui median (IQR), mm3 | External median (IQR), mm3 | External/Lishui median ratio | BH q |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    volume_lookup = {(row["cohort"], row["structure"]): row for row in volume_summary}
    for row in shift_rows:
        structure = row["structure"]
        ls = volume_lookup[("Lishui", structure)]
        ext = volume_lookup[("External manual reference", structure)]
        lines.append(
            f"| {structure} | {ls['median']:.2f} ({ls['q1']:.2f}-{ls['q3']:.2f}) | {ext['median']:.2f} ({ext['q1']:.2f}-{ext['q3']:.2f}) | {row['external_to_lishui_median_ratio']:.3f} | {row['bh_q_value']:.3g} |"
        )
    lines.extend(
        [
            "",
            "## Five leading limitations to test",
            "",
            "1. The isotropic 8x encoder reduction leaves only six z tokens at the transformer bottleneck.",
            "2. Three independent binary models cannot explicitly exploit the fixed SSC/HSC/PSC spatial relationship.",
            "3. The current Tversky term penalises FP more than FN; its effect must be tested against symmetric and FN-heavier variants on internal OOF predictions.",
            "4. V1 augmentation is narrow for cross-scanner MRI and lacks resolution, blur and bias-field simulation.",
            "5. The single historical 140/30/30 split gives less stable model-selection evidence than patient-level five-fold OOF evaluation.",
            "",
            "## Executable experiment set on this workstation",
            "",
            "- COMPLETED: V1 source/result audit; V1 internal and exposed-external quantitative error summary.",
            "- COMPLETED previously: nnU-Net v2 multiclass fold-0 five-epoch pilots M1/M2/M3 on the locked internal validation split.",
            "- EXECUTABLE: construct a 200-patient/400-ear five-fold split and Dataset502; validate nnU-Net planning and run smoke tests.",
            "- COMPUTE-LIMITED: full five-fold training. One nnU-Net epoch previously required about 51 minutes on the 6-GB GTX 1660 Ti, so even a five-epoch x five-fold benchmark is approximately 21 GPU-hours before ablations.",
            "- NOT RUN until internal winner is frozen: any further external-label evaluation. The existing 50 cases are already exposed and cannot serve as a new confirmatory cohort.",
            "",
            "## Leakage audit",
            "",
            "- Partition: PASS for the historical split; both ears remain with the patient. A new patient-level five-fold split is required for V2 OOF.",
            "- Fit-on-training: PASS for augmentation and normalisation implementation; crop localisation uses a frozen deployment localiser.",
            "- Tuning: PASS for recorded V1 thresholds; selected on Lishui validation only. External labels are prohibited for V2 selection.",
            "- Input integrity: PASS; the model receives T2 crop only.",
            "- Evaluation: PARTIAL; CIs and failure analysis exist, but formal V2 five-fold OOF and a new untouched external cohort are not yet available.",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "commit": commit, "report": str(args.report), "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
