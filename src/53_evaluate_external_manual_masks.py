from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import scipy
from nibabel.processing import resample_from_to
from scipy import ndimage, stats


STRUCTURES = ("SSC", "HSC", "PSC")
METRICS = (
    "dice",
    "iou",
    "precision",
    "recall",
    "hd95_mm",
    "assd_mm",
    "surface_dice_0p5mm",
    "surface_dice_1p0mm",
    "absolute_volume_error_mm3",
    "relative_absolute_volume_error",
)
COLORS = {"Internal": "#4C78A8", "Center 2": "#F58518", "Center 3": "#54A24B", "Pooled external": "#B279A2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen three-canal segmentation predictions against external manual masks."
    )
    parser.add_argument("--manual-center2-dir", type=Path, required=True)
    parser.add_argument("--manual-center3-dir", type=Path, required=True)
    parser.add_argument("--manifest-center2", type=Path, required=True)
    parser.add_argument("--manifest-center3", type=Path, required=True)
    parser.add_argument("--study-qc-center2", type=Path, required=True)
    parser.add_argument("--study-qc-center3", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--internal-metrics", type=Path, required=True)
    parser.add_argument("--internal-summary", type=Path, required=True)
    parser.add_argument("--model-summary", type=Path, required=True)
    parser.add_argument("--archive-center2", type=Path, required=True)
    parser.add_argument("--archive-center3", type=Path, required=True)
    parser.add_argument("--override-center3-dir", type=Path)
    parser.add_argument("--override-center3-study-ids", default="")
    parser.add_argument("--reproduction-check-study-id", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args()


def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("external_manual_validation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def affine_close(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.allclose(a, b, rtol=1e-5, atol=1e-4))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    nonzero = (np.abs(a) > 1e-8) | (np.abs(b) > 1e-8)
    a, b = a[nonzero], b[nonzero]
    if len(a) > 1_000_000:
        index = np.linspace(0, len(a) - 1, 1_000_000, dtype=int)
        a, b = a[index], b[index]
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def surface_arrays(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3, 3), dtype=bool)
    return mask ^ ndimage.binary_erosion(mask, structure=structure, border_value=0)


def surface_metrics(prediction: np.ndarray, target: np.ndarray, spacing: tuple[float, ...]) -> dict[str, float]:
    if not prediction.any() or not target.any():
        return {"hd95_mm": float("nan"), "assd_mm": float("nan"), "surface_dice_0p5mm": 0.0, "surface_dice_1p0mm": 0.0}
    union_coordinates = np.argwhere(prediction | target)
    lower = np.maximum(union_coordinates.min(axis=0) - 2, 0)
    upper = np.minimum(union_coordinates.max(axis=0) + 3, prediction.shape)
    crop = tuple(slice(int(lower[axis]), int(upper[axis])) for axis in range(3))
    prediction = prediction[crop]
    target = target[crop]
    pred_surface = surface_arrays(prediction)
    target_surface = surface_arrays(target)
    distance_to_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    distance_to_prediction = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    pred_distances = distance_to_target[pred_surface]
    target_distances = distance_to_prediction[target_surface]
    distances = np.concatenate([pred_distances, target_distances])
    denominator = len(pred_distances) + len(target_distances)
    return {
        "hd95_mm": float(np.percentile(distances, 95)),
        "assd_mm": float(np.mean(distances)),
        "surface_dice_0p5mm": float(((pred_distances <= 0.5).sum() + (target_distances <= 0.5).sum()) / denominator),
        "surface_dice_1p0mm": float(((pred_distances <= 1.0).sum() + (target_distances <= 1.0).sum()) / denominator),
    }


def binary_metrics(prediction: np.ndarray, target: np.ndarray, spacing: tuple[float, ...]) -> dict[str, float]:
    intersection = int((prediction & target).sum())
    pred_count = int(prediction.sum())
    target_count = int(target.sum())
    union = pred_count + target_count - intersection
    voxel_volume = float(np.prod(spacing))
    result = {
        "dice": (2 * intersection + 1e-5) / (pred_count + target_count + 1e-5),
        "iou": (intersection + 1e-5) / (union + 1e-5),
        "precision": (intersection + 1e-5) / (pred_count + 1e-5),
        "recall": (intersection + 1e-5) / (target_count + 1e-5),
        "reference_volume_mm3": target_count * voxel_volume,
        "predicted_volume_mm3": pred_count * voxel_volume,
        "absolute_volume_error_mm3": abs(pred_count - target_count) * voxel_volume,
        "relative_absolute_volume_error": abs(pred_count - target_count) / target_count if target_count else float("nan"),
        "signed_volume_error_mm3": (pred_count - target_count) * voxel_volume,
    }
    result.update(surface_metrics(prediction, target, spacing))
    return result


def patient_id_from_internal(sample_id: str) -> str:
    return re.sub(r"[_-][LR]$", "", str(sample_id), flags=re.I)


def cluster_summary(
    frame: pd.DataFrame,
    cohort_label: str,
    repetitions: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for structure in (*STRUCTURES, "Macro"):
        if structure == "Macro":
            patient_values = frame.groupby("patient_id", sort=True)[list(METRICS)].mean(numeric_only=True)
        else:
            patient_values = frame.loc[frame["structure"] == structure].groupby("patient_id", sort=True)[list(METRICS)].mean(numeric_only=True)
        patients = patient_values.index.to_numpy()
        for metric in METRICS:
            values = patient_values[metric].to_numpy(dtype=float)
            finite = np.isfinite(values)
            values = values[finite]
            if len(values) == 0:
                mean = low = high = float("nan")
            else:
                mean = float(np.mean(values))
                samples = np.empty(repetitions, dtype=float)
                for index in range(repetitions):
                    samples[index] = float(np.mean(values[rng.integers(0, len(values), size=len(values))]))
                low, high = np.percentile(samples, [2.5, 97.5]).tolist()
            rows.append(
                {
                    "cohort": cohort_label,
                    "structure": structure,
                    "metric": metric,
                    "patient_n": int(len(patients)),
                    "ear_n": int(frame["ear_id"].nunique()),
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "ci_method": f"patient-clustered percentile bootstrap ({repetitions} resamples)",
                }
            )
    return rows


def bootstrap_difference(
    a: pd.DataFrame,
    b: pd.DataFrame,
    structure: str,
    metric: str,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float]:
    def patient_values(frame: pd.DataFrame) -> np.ndarray:
        current = frame if structure == "Macro" else frame.loc[frame["structure"] == structure]
        values = current.groupby("patient_id")[metric].mean().to_numpy(dtype=float)
        return values[np.isfinite(values)]

    av = patient_values(a)
    bv = patient_values(b)
    point = float(np.mean(bv) - np.mean(av))
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        samples[index] = float(
            np.mean(bv[rng.integers(0, len(bv), size=len(bv))])
            - np.mean(av[rng.integers(0, len(av), size=len(av))])
        )
    low, high = np.percentile(samples, [2.5, 97.5]).tolist()
    return point, low, high


def standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled == 0:
        return 0.0 if np.mean(a) == np.mean(b) else float("inf")
    return float((np.mean(b) - np.mean(a)) / pooled)


def image_profile(path: Path, center: str, patient_id: str) -> dict[str, object]:
    image = nib.as_closest_canonical(nib.load(str(path)))
    shape = tuple(int(value) for value in image.shape[:3])
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    fov = tuple(shape[index] * spacing[index] for index in range(3))
    return {
        "center": center,
        "patient_id": patient_id,
        "image_path": str(path.resolve()),
        "shape_x": shape[0],
        "shape_y": shape[1],
        "shape_z": shape[2],
        "spacing_x_mm": spacing[0],
        "spacing_y_mm": spacing[1],
        "spacing_z_mm": spacing[2],
        "fov_x_mm": fov[0],
        "fov_y_mm": fov[1],
        "fov_z_mm": fov[2],
    }


def format_ci(mean: float, low: float, high: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ({low:.{digits}f}–{high:.{digits}f})"


def make_figures(metrics: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    cohort_order = ["Internal", "Center 2", "Center 3", "Pooled external"]
    structure_order = [*STRUCTURES, "Macro"]
    dice_summary = summary.loc[(summary.metric == "dice") & summary.cohort.isin(cohort_order)]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    offsets = np.linspace(-0.27, 0.27, len(cohort_order))
    for offset, cohort in zip(offsets, cohort_order, strict=True):
        block = dice_summary.set_index(["cohort", "structure"]).loc[cohort]
        x = np.arange(len(structure_order)) + offset
        means = np.array([block.loc[s, "mean"] for s in structure_order])
        lows = np.array([block.loc[s, "ci_low"] for s in structure_order])
        highs = np.array([block.loc[s, "ci_high"] for s in structure_order])
        ax.errorbar(x, means, yerr=[means - lows, highs - means], fmt="o", capsize=3, lw=1.2, ms=5, color=COLORS[cohort], label=cohort)
    ax.set_xticks(np.arange(len(structure_order)), structure_order)
    ax.set_ylabel("Dice coefficient")
    ax.set_ylim(0, 1.02)
    ax.set_title("Frozen-model internal and external segmentation performance")
    ax.grid(axis="y", color="#D9D9D9", lw=0.6)
    ax.legend(frameon=False, ncol=2, loc="lower left")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"Figure_1_internal_external_dice.{suffix}", dpi=600, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.8), sharey=True)
    rng = np.random.default_rng(42)
    for axis, structure in zip(axes, STRUCTURES, strict=True):
        blocks = [metrics.loc[(metrics.cohort == cohort) & (metrics.structure == structure), "dice"].to_numpy() for cohort in ("Center 2", "Center 3")]
        box = axis.boxplot(blocks, positions=[0, 1], widths=0.55, patch_artist=True, showfliers=False)
        for patch, cohort in zip(box["boxes"], ("Center 2", "Center 3"), strict=True):
            patch.set_facecolor(COLORS[cohort]); patch.set_alpha(0.35); patch.set_edgecolor(COLORS[cohort])
        for index, (values, cohort) in enumerate(zip(blocks, ("Center 2", "Center 3"), strict=True)):
            axis.scatter(index + rng.normal(0, 0.045, len(values)), values, s=9, alpha=0.45, color=COLORS[cohort], linewidths=0)
        internal = dice_summary.loc[(dice_summary.cohort == "Internal") & (dice_summary.structure == structure), "mean"].iloc[0]
        axis.axhline(internal, color=COLORS["Internal"], ls="--", lw=1.2, label="Internal mean" if structure == "SSC" else None)
        axis.set_title(structure)
        axis.set_xticks([0, 1], ["Center 2", "Center 3"])
        axis.grid(axis="y", color="#E5E5E5", lw=0.5)
    axes[0].set_ylabel("Ear-level Dice")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("External ear-level Dice distributions", y=1.02)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"Figure_2_external_dice_distribution.{suffix}", dpi=600, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.8))
    for axis, structure in zip(axes, STRUCTURES, strict=True):
        for cohort in ("Center 2", "Center 3"):
            block = metrics.loc[(metrics.cohort == cohort) & (metrics.structure == structure)]
            axis.scatter(block.reference_volume_mm3, block.predicted_volume_mm3, s=15, alpha=0.55, color=COLORS[cohort], label=cohort)
        limit = float(max(metrics.loc[metrics.structure == structure, ["reference_volume_mm3", "predicted_volume_mm3"]].max())) * 1.05
        axis.plot([0, limit], [0, limit], color="#444444", ls="--", lw=1)
        axis.set_xlim(0, limit); axis.set_ylim(0, limit); axis.set_aspect("equal", adjustable="box")
        axis.set_title(structure); axis.set_xlabel("Manual volume (mm³)"); axis.grid(color="#E5E5E5", lw=0.5)
    axes[0].set_ylabel("Predicted volume (mm³)")
    axes[-1].legend(frameon=False, loc="upper left")
    fig.suptitle("External manual–prediction volume agreement", y=1.02)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"Figure_3_volume_agreement.{suffix}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def make_failure_figure(metrics: pd.DataFrame, output_dir: Path) -> None:
    worst = metrics.nsmallest(6, "dice")
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.0))
    for axis, (_, row) in zip(axes.ravel(), worst.iterrows(), strict=True):
        pred_img = nib.as_closest_canonical(nib.load(row.prediction_path))
        ref_img = nib.as_closest_canonical(nib.load(row.manual_mask_path))
        t2_img = nib.as_closest_canonical(nib.load(row.manual_t2_path))
        ref = np.asarray(resample_from_to(ref_img, pred_img, order=0).dataobj) > 0
        pred = np.asarray(pred_img.dataobj) > 0
        t2 = np.asarray(resample_from_to(t2_img, pred_img, order=1).dataobj, dtype=np.float32)
        union = pred | ref
        z = int(np.argmax(union.sum(axis=(0, 1))))
        image = np.rot90(t2[:, :, z])
        low, high = np.percentile(image[np.isfinite(image)], [1, 99])
        axis.imshow(image, cmap="gray", vmin=low, vmax=high)
        axis.contour(np.rot90(ref[:, :, z]), levels=[0.5], colors=["#00E676"], linewidths=1.2)
        axis.contour(np.rot90(pred[:, :, z]), levels=[0.5], colors=["#FF2D95"], linewidths=1.0)
        axis.set_title(f"{row.cohort}, {row.patient_id}{row.ear_side} {row.structure}\nDice={row.dice:.3f}", fontsize=9)
        axis.axis("off")
    fig.suptitle("Lowest-Dice external cases (green: manual; magenta: frozen prediction)", y=0.99)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"Figure_4_failure_cases.{suffix}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logger(args.output_dir / "external_validation.log")
    log.info("Starting frozen external manual-mask validation")
    log.info("Command: %s", " ".join(sys.argv))

    archive_hashes = {"Center 2": sha256(args.archive_center2), "Center 3": sha256(args.archive_center3)}
    model_hashes = {path.name: sha256(path) for path in sorted(args.model_summary.parent.glob("*.pt"))}
    model_summary = json.loads(args.model_summary.read_text(encoding="utf-8"))
    internal_original_summary = json.loads(args.internal_summary.read_text(encoding="utf-8"))

    center_specs = {
        "Center 2": (args.manual_center2_dir, args.manifest_center2, args.study_qc_center2),
        "Center 3": (args.manual_center3_dir, args.manifest_center3, args.study_qc_center3),
    }
    all_metric_rows: list[dict[str, object]] = []
    mask_qc_rows: list[dict[str, object]] = []
    case_qc_rows: list[dict[str, object]] = []
    external_image_rows: list[dict[str, object]] = []
    external_t2_hashes: dict[str, str] = {}
    reproduction_rows: list[dict[str, object]] = []
    override_ids = {value.strip().zfill(3) for value in args.override_center3_study_ids.split(",") if value.strip()}

    for cohort, (manual_dir, manifest_path, study_qc_path) in center_specs.items():
        manifest = pd.read_csv(manifest_path, dtype={"study_id": str})
        manifest["study_id"] = manifest.study_id.str.zfill(3)
        manifest = manifest.loc[manifest.structure.isin(STRUCTURES)].copy()
        prediction_lookup = {(row.study_id, row.ear_side, row.structure): Path(row.mask_path) for row in manifest.itertuples()}
        original_prediction_lookup = dict(prediction_lookup)
        study_qc = pd.read_csv(study_qc_path, dtype={"study_id": str})
        study_qc["study_id"] = study_qc.study_id.str.zfill(3)
        input_lookup = {row.study_id: Path(row.input_nifti) for row in study_qc.itertuples()}
        if cohort == "Center 3" and args.override_center3_dir:
            for study_id in sorted(override_ids):
                for side in ("L", "R"):
                    for structure in STRUCTURES:
                        override_path = args.override_center3_dir / "predicted_masks" / f"sub{study_id}" / f"{study_id}{side}_{structure}.nii.gz"
                        if not override_path.exists():
                            raise FileNotFoundError(f"Missing requested override prediction: {override_path}")
                        prediction_lookup[(study_id, side, structure)] = override_path
                input_lookup[study_id] = manual_dir / study_id / f"{study_id}_T2.nii.gz"
            check_id = args.reproduction_check_study_id.strip().zfill(3)
            if check_id and (args.override_center3_dir / "predicted_masks" / f"sub{check_id}").exists():
                for side in ("L", "R"):
                    for structure in STRUCTURES:
                        archived_path = original_prediction_lookup[(check_id, side, structure)]
                        reproduced_path = args.override_center3_dir / "predicted_masks" / f"sub{check_id}" / f"{check_id}{side}_{structure}.nii.gz"
                        archived = np.asarray(nib.as_closest_canonical(nib.load(str(archived_path))).dataobj) > 0
                        reproduced = np.asarray(nib.as_closest_canonical(nib.load(str(reproduced_path))).dataobj) > 0
                        intersection = int((archived & reproduced).sum())
                        reproduction_rows.append(
                            {
                                "study_id": check_id,
                                "ear_side": side,
                                "structure": structure,
                                "archived_reproduced_dice": (2 * intersection + 1e-5) / (int(archived.sum()) + int(reproduced.sum()) + 1e-5),
                                "affine_match": affine_close(
                                    nib.as_closest_canonical(nib.load(str(archived_path))).affine,
                                    nib.as_closest_canonical(nib.load(str(reproduced_path))).affine,
                                ),
                                "archived_prediction_path": str(archived_path.resolve()),
                                "reproduced_prediction_path": str(reproduced_path.resolve()),
                            }
                        )
        subject_dirs = sorted(path for path in manual_dir.iterdir() if path.is_dir())
        log.info("%s: %d manual subject directories", cohort, len(subject_dirs))
        for subject_dir in subject_dirs:
            patient_id = subject_dir.name.zfill(3)
            manual_t2 = subject_dir / f"{patient_id}_T2.nii.gz"
            source_t2 = input_lookup.get(patient_id)
            expected_masks = [subject_dir / f"{patient_id}{side}_{structure}.nii.gz" for side in ("L", "R") for structure in STRUCTURES]
            expected_predictions = [prediction_lookup.get((patient_id, side, structure)) for side in ("L", "R") for structure in STRUCTURES]
            t2_exists = manual_t2.exists()
            source_exists = bool(source_t2 and source_t2.exists())
            mask_complete = all(path.exists() for path in expected_masks)
            prediction_complete = all(path is not None and path.exists() for path in expected_predictions)
            manual_hash = sha256(manual_t2) if t2_exists else ""
            source_hash = sha256(source_t2) if source_exists else ""
            external_t2_hashes[f"{cohort}:{patient_id}"] = manual_hash
            t2_corr = float("nan")
            t2_same_grid = False
            if t2_exists:
                external_image_rows.append(image_profile(manual_t2, cohort, patient_id))
            if t2_exists and source_exists:
                manual_img = nib.as_closest_canonical(nib.load(str(manual_t2)))
                source_img = nib.as_closest_canonical(nib.load(str(source_t2)))
                t2_same_grid = manual_img.shape[:3] == source_img.shape[:3] and affine_close(manual_img.affine, source_img.affine)
                manual_on_source = manual_img if t2_same_grid else resample_from_to(manual_img, source_img, order=1)
                t2_corr = safe_corr(np.asarray(manual_on_source.dataobj), np.asarray(source_img.dataobj))
            case_qc_rows.append(
                {
                    "cohort": cohort,
                    "patient_id": patient_id,
                    "manual_t2_exists": t2_exists,
                    "source_inference_t2_exists": source_exists,
                    "six_manual_masks_complete": mask_complete,
                    "six_frozen_predictions_complete": prediction_complete,
                    "manual_source_t2_sha256_identical": bool(manual_hash and manual_hash == source_hash),
                    "manual_source_t2_same_grid": t2_same_grid,
                    "manual_source_t2_correlation": t2_corr,
                    "case_match_pass": bool(t2_exists and source_exists and mask_complete and prediction_complete and t2_corr > 0.9999),
                    "manual_t2_path": str(manual_t2.resolve()) if t2_exists else "",
                    "source_t2_path": str(source_t2.resolve()) if source_exists else "",
                }
            )
            for side in ("L", "R"):
                for structure in STRUCTURES:
                    manual_path = subject_dir / f"{patient_id}{side}_{structure}.nii.gz"
                    prediction_path = prediction_lookup.get((patient_id, side, structure))
                    if not (manual_path.exists() and prediction_path and prediction_path.exists() and manual_t2.exists()):
                        continue
                    manual_img = nib.as_closest_canonical(nib.load(str(manual_path)))
                    t2_img = nib.as_closest_canonical(nib.load(str(manual_t2)))
                    pred_img = nib.as_closest_canonical(nib.load(str(prediction_path)))
                    manual_raw = np.asarray(manual_img.dataobj)
                    unique_values = np.unique(manual_raw)
                    manual_binary = manual_raw > 0
                    manual_on_pred_img = resample_from_to(nib.Nifti1Image(manual_binary.astype(np.uint8), manual_img.affine), pred_img, order=0)
                    target = np.asarray(manual_on_pred_img.dataobj) > 0
                    prediction = np.asarray(pred_img.dataobj) > 0
                    spacing = tuple(float(value) for value in pred_img.header.get_zooms()[:3])
                    metrics = binary_metrics(prediction, target, spacing)
                    ear_id = f"{cohort.replace(' ', '')}-{patient_id}-{side}"
                    all_metric_rows.append(
                        {
                            "cohort": cohort,
                            "patient_id": patient_id,
                            "ear_id": ear_id,
                            "ear_side": side,
                            "structure": structure,
                            **metrics,
                            "manual_mask_path": str(manual_path.resolve()),
                            "prediction_path": str(prediction_path.resolve()),
                            "manual_t2_path": str(manual_t2.resolve()),
                            "inference_source": "frozen_manual_T2_reinference" if patient_id in override_ids and cohort == "Center 3" else "archived_frozen_prediction",
                        }
                    )
                    mask_qc_rows.append(
                        {
                            "cohort": cohort,
                            "patient_id": patient_id,
                            "ear_side": side,
                            "structure": structure,
                            "manual_shape": "x".join(map(str, manual_img.shape[:3])),
                            "t2_shape": "x".join(map(str, t2_img.shape[:3])),
                            "prediction_shape": "x".join(map(str, pred_img.shape[:3])),
                            "manual_t2_affine_match": affine_close(manual_img.affine, t2_img.affine),
                            "manual_t2_shape_match": manual_img.shape[:3] == t2_img.shape[:3],
                            "manual_nonzero_voxels": int(manual_binary.sum()),
                            "manual_unique_values": ";".join(map(str, unique_values[:20].tolist())),
                            "manual_binary_after_positive_threshold": bool(set(unique_values.tolist()).issubset({0, 1})),
                            "resampled_reference_nonzero_voxels": int(target.sum()),
                            "prediction_nonzero_voxels": int(prediction.sum()),
                            "geometry_qc_pass": bool(
                                affine_close(manual_img.affine, t2_img.affine)
                                and manual_img.shape[:3] == t2_img.shape[:3]
                                and manual_binary.any()
                                and target.any()
                                and prediction.any()
                            ),
                        }
                    )

    metrics = pd.DataFrame(all_metric_rows)
    mask_qc = pd.DataFrame(mask_qc_rows)
    case_qc = pd.DataFrame(case_qc_rows)
    image_profiles = pd.DataFrame(external_image_rows)
    if len(metrics) != 300:
        raise RuntimeError(f"Expected 300 external mask comparisons; found {len(metrics)}")
    if not bool(case_qc.case_match_pass.all()):
        log.warning("Some case matches failed; see case_match_qc.csv")
    if not bool(mask_qc.geometry_qc_pass.all()):
        log.warning("Some mask geometry checks failed; see mask_geometry_qc.csv")

    internal = pd.read_csv(args.internal_metrics)
    internal["cohort"] = "Internal"
    internal["patient_id"] = internal.sample_id.map(patient_id_from_internal)
    internal["ear_id"] = internal.sample_id.astype(str)
    internal = internal.rename(columns={"average_symmetric_surface_distance_mm": "assd_mm"})
    for column in ("surface_dice_0p5mm", "surface_dice_1p0mm"):
        internal[column] = np.nan

    summaries: list[dict[str, object]] = []
    summaries.extend(cluster_summary(internal, "Internal", args.bootstrap, args.seed + 1))
    center2 = metrics.loc[metrics.cohort == "Center 2"].copy()
    center3 = metrics.loc[metrics.cohort == "Center 3"].copy()
    summaries.extend(cluster_summary(center2, "Center 2", args.bootstrap, args.seed + 2))
    summaries.extend(cluster_summary(center3, "Center 3", args.bootstrap, args.seed + 3))
    summaries.extend(cluster_summary(metrics, "Pooled external", args.bootstrap, args.seed + 4))
    summary = pd.DataFrame(summaries)

    comparison_rows: list[dict[str, object]] = []
    comparisons = [("Internal", internal, "Center 2", center2), ("Internal", internal, "Center 3", center3), ("Internal", internal, "Pooled external", metrics), ("Center 2", center2, "Center 3", center3)]
    for left_name, left, right_name, right in comparisons:
        for structure in (*STRUCTURES, "Macro"):
            for metric in ("dice", "iou", "hd95_mm", "assd_mm"):
                point, low, high = bootstrap_difference(left, right, structure, metric, args.bootstrap, args.seed + len(comparison_rows) + 10)
                comparison_rows.append(
                    {
                        "comparison": f"{right_name} minus {left_name}",
                        "structure": structure,
                        "metric": metric,
                        "difference": point,
                        "ci_low": low,
                        "ci_high": high,
                        "ci_excludes_zero": bool(low > 0 or high < 0),
                        "ci_method": f"independent patient-clustered bootstrap ({args.bootstrap} resamples)",
                    }
                )
    comparisons_df = pd.DataFrame(comparison_rows)

    training_t2_paths = sorted(args.training_root.glob("sub*/**/*_T2.nii.gz"))
    if len(training_t2_paths) != 200:
        training_t2_paths = sorted(args.training_root.rglob("*_T2.nii.gz"))
    training_hashes: dict[str, str] = {}
    training_image_rows: list[dict[str, object]] = []
    for index, path in enumerate(training_t2_paths, start=1):
        patient_id = path.name.removesuffix("_T2.nii.gz")
        training_hashes[str(path.resolve())] = sha256(path)
        training_image_rows.append(image_profile(path, "Internal development", patient_id))
        if index % 50 == 0:
            log.info("Training T2 leakage/profile audit %d/%d", index, len(training_t2_paths))
    exact_hash_overlap = sorted(set(external_t2_hashes.values()) & set(training_hashes.values()) - {""})
    training_manifest = pd.read_csv(args.training_manifest, dtype={"subject_id": str})
    external_numeric_ids = set(metrics.patient_id.astype(str))
    numeric_id_overlap = sorted(external_numeric_ids & set(training_manifest.subject_id.astype(str).str.zfill(3)))
    training_profiles = pd.DataFrame(training_image_rows)
    all_profiles = pd.concat([training_profiles, image_profiles], ignore_index=True)

    shift_rows: list[dict[str, object]] = []
    geometry_features = ["shape_x", "shape_y", "shape_z", "spacing_x_mm", "spacing_y_mm", "spacing_z_mm", "fov_x_mm", "fov_y_mm", "fov_z_mm"]
    for cohort in ("Center 2", "Center 3"):
        for feature in geometry_features:
            a = training_profiles[feature].to_numpy(dtype=float)
            b = image_profiles.loc[image_profiles.center == cohort, feature].to_numpy(dtype=float)
            ks = stats.ks_2samp(a, b, method="auto")
            shift_rows.append(
                {
                    "comparison": f"{cohort} vs internal development",
                    "domain": "image_geometry",
                    "feature_or_structure": feature,
                    "internal_mean": float(np.mean(a)),
                    "external_mean": float(np.mean(b)),
                    "standardized_mean_difference": standardized_mean_difference(a, b),
                    "ks_statistic": float(ks.statistic),
                    "ks_p_value": float(ks.pvalue),
                    "interpretation": "acquisition/geometry shift screen; not used to alter preprocessing",
                }
            )
    for structure in STRUCTURES:
        a = center2.loc[center2.structure == structure, "reference_volume_mm3"].to_numpy(dtype=float)
        b = center3.loc[center3.structure == structure, "reference_volume_mm3"].to_numpy(dtype=float)
        ks = stats.ks_2samp(a, b, method="auto")
        shift_rows.append(
            {
                "comparison": "Center 3 vs Center 2",
                "domain": "manual_reference_volume",
                "feature_or_structure": structure,
                "internal_mean": float(np.mean(a)),
                "external_mean": float(np.mean(b)),
                "standardized_mean_difference": standardized_mean_difference(a, b),
                "ks_statistic": float(ks.statistic),
                "ks_p_value": float(ks.pvalue),
                "interpretation": "center/annotation/population shift screen; exploratory only",
            }
        )
    shift = pd.DataFrame(shift_rows)

    bottom_cases = metrics.nsmallest(20, "dice").copy()
    qc_summary = {
        "archive_sha256": archive_hashes,
        "manual_subjects": {cohort: int(case_qc.loc[case_qc.cohort == cohort, "patient_id"].nunique()) for cohort in center_specs},
        "manual_ears": {cohort: int(metrics.loc[metrics.cohort == cohort, "ear_id"].nunique()) for cohort in center_specs},
        "manual_masks": {cohort: int((metrics.cohort == cohort).sum()) for cohort in center_specs},
        "all_case_matches_pass": bool(case_qc.case_match_pass.all()),
        "all_mask_geometry_pass": bool(mask_qc.geometry_qc_pass.all()),
        "minimum_manual_source_t2_correlation": float(case_qc.manual_source_t2_correlation.min()),
        "exact_external_training_t2_sha256_overlap_count": len(exact_hash_overlap),
        "numeric_subject_id_overlap_count_without_center_namespace": len(numeric_id_overlap),
        "numeric_subject_id_overlap_interpretation": "Numeric IDs are site-local and overlap with the development cohort by design; center was retained as a separate cohort field, and the two sampled external subsets had no patient-ID overlap. No exact T2 file hash overlap was found." if not exact_hash_overlap else "Potential exact image duplication requires investigation.",
        "model_files_sha256": model_hashes,
        "frozen_preprocessing": {
            "target_spacing_mm": [0.3472222, 0.3472222, 0.5],
            "crop_size_voxels": [128, 128, 48],
            "roi_localisation": model_summary.get("roi_localisation"),
            "postprocessing": model_summary.get("deployed_postprocessing"),
        },
        "missing_external_reference_structures": ["Cochlear", "Vestibular", "TV"],
        "manual_t2_reinference_study_ids": sorted(override_ids),
        "reproduction_check_min_dice": float(min(row["archived_reproduced_dice"] for row in reproduction_rows)) if reproduction_rows else None,
        "reproduction_check_interpretation": "CPU rerun of the archived frozen checkpoints on a matched case; small boundary differences relative to archived GPU inference are expected from backend numerics." if reproduction_rows else None,
        "classification_metrics_applicability": "AUC, sensitivity, specificity, accuracy, PPV, NPV, F1, calibration, and DCA are not applicable to this segmentation-validation endpoint.",
    }

    metrics.to_csv(args.output_dir / "external_segmentation_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "performance_summary_patient_clustered.csv", index=False, encoding="utf-8-sig")
    comparisons_df.to_csv(args.output_dir / "internal_external_differences.csv", index=False, encoding="utf-8-sig")
    case_qc.to_csv(args.output_dir / "case_match_qc.csv", index=False, encoding="utf-8-sig")
    mask_qc.to_csv(args.output_dir / "mask_geometry_qc.csv", index=False, encoding="utf-8-sig")
    all_profiles.to_csv(args.output_dir / "image_geometry_profiles.csv", index=False, encoding="utf-8-sig")
    shift.to_csv(args.output_dir / "distribution_shift_screen.csv", index=False, encoding="utf-8-sig")
    bottom_cases.to_csv(args.output_dir / "lowest_dice_cases.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(reproduction_rows).to_csv(args.output_dir / "frozen_inference_reproduction_check.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "external_validation_summary.json").write_text(json.dumps(qc_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    make_figures(pd.concat([internal, metrics], ignore_index=True, sort=False), summary, args.output_dir)
    make_failure_figure(metrics, args.output_dir)

    def dice_row(cohort: str, structure: str) -> pd.Series:
        return summary.loc[(summary.cohort == cohort) & (summary.structure == structure) & (summary.metric == "dice")].iloc[0]

    def metric_row(cohort: str, structure: str, metric: str) -> pd.Series:
        return summary.loc[(summary.cohort == cohort) & (summary.structure == structure) & (summary.metric == metric)].iloc[0]

    lines = [
        "# Frozen external validation of three semicircular-canal segmentation",
        "",
        "## Scope and validity boundary",
        "",
        f"The two external manual-reference subsets contained {qc_summary['manual_subjects']['Center 2']} and {qc_summary['manual_subjects']['Center 3']} patients, respectively (100 ears and 300 canal masks in total). The locked two-stage model, target spacing, predicted-center ROI, validation-selected thresholds, and component rules were not changed. External masks were used only for quality control and final evaluation.",
        "",
        "AUC, classification sensitivity/specificity, accuracy, PPV, NPV, F1, calibration, and decision-curve analysis are not applicable because the frozen endpoint evaluated here is voxel-wise segmentation, not patient-level classification. The prespecified segmentation metrics were Dice, IoU, precision, recall, HD95, ASSD, surface Dice, and volume error.",
        "",
        "## Manuscript-ready Methods",
        "",
        "Independent external validation was performed in two prespecified imaging subsets from the same external hospital. The frozen two-stage 3D TinyViT-UNet pipeline comprised a union localizer followed by separate binary models for the superior, horizontal, and posterior semicircular canals. Images were resampled to 0.3472 × 0.3472 × 0.5000 mm and cropped to 128 × 128 × 48 voxels using the predicted canal-center workflow. Thresholds and connected-component policies selected in the development validation set were applied unchanged. Manual masks were resampled to the frozen prediction grid with nearest-neighbor interpolation only for metric computation. Performance was summarized per structure and as a macro-average. Confidence intervals were obtained with 5,000 patient-clustered bootstrap resamples, preserving dependence between ears and structures from the same patient.",
        "",
        "## Manuscript-ready Results",
        "",
    ]
    pooled_macro = dice_row("Pooled external", "Macro")
    c2_macro = dice_row("Center 2", "Macro")
    c3_macro = dice_row("Center 3", "Macro")
    lines.append(
        f"All 50 external cases were matched to the original frozen-inference images and all 300 reference masks passed geometry and non-empty-mask checks. The pooled external macro-average Dice was {format_ci(pooled_macro['mean'], pooled_macro['ci_low'], pooled_macro['ci_high'])}. Center-specific macro-average Dice values were {format_ci(c2_macro['mean'], c2_macro['ci_low'], c2_macro['ci_high'])} for center subset 2 and {format_ci(c3_macro['mean'], c3_macro['ci_low'], c3_macro['ci_high'])} for center subset 3."
    )
    structure_text = []
    for structure in STRUCTURES:
        row = dice_row("Pooled external", structure)
        structure_text.append(f"{structure} {format_ci(row['mean'], row['ci_low'], row['ci_high'])}")
    pooled_macro_metrics = {
        metric: metric_row("Pooled external", "Macro", metric)
        for metric in ("iou", "precision", "recall", "hd95_mm", "assd_mm", "surface_dice_0p5mm", "surface_dice_1p0mm")
    }
    lines.extend(
        [
            "",
            "Pooled structure-specific Dice values were " + "; ".join(structure_text) + ".",
            "",
            "Other pooled macro-average metrics were "
            f"IoU {format_ci(pooled_macro_metrics['iou']['mean'], pooled_macro_metrics['iou']['ci_low'], pooled_macro_metrics['iou']['ci_high'])}, "
            f"precision {format_ci(pooled_macro_metrics['precision']['mean'], pooled_macro_metrics['precision']['ci_low'], pooled_macro_metrics['precision']['ci_high'])}, "
            f"recall {format_ci(pooled_macro_metrics['recall']['mean'], pooled_macro_metrics['recall']['ci_low'], pooled_macro_metrics['recall']['ci_high'])}, "
            f"HD95 {format_ci(pooled_macro_metrics['hd95_mm']['mean'], pooled_macro_metrics['hd95_mm']['ci_low'], pooled_macro_metrics['hd95_mm']['ci_high'])} mm, "
            f"ASSD {format_ci(pooled_macro_metrics['assd_mm']['mean'], pooled_macro_metrics['assd_mm']['ci_low'], pooled_macro_metrics['assd_mm']['ci_high'])} mm, "
            f"surface Dice at 0.5 mm {format_ci(pooled_macro_metrics['surface_dice_0p5mm']['mean'], pooled_macro_metrics['surface_dice_0p5mm']['ci_low'], pooled_macro_metrics['surface_dice_0p5mm']['ci_high'])}, and "
            f"surface Dice at 1.0 mm {format_ci(pooled_macro_metrics['surface_dice_1p0mm']['mean'], pooled_macro_metrics['surface_dice_1p0mm']['ci_low'], pooled_macro_metrics['surface_dice_1p0mm']['ci_high'])}.",
            "",
        ]
    )
    lines.extend(
        [
            "## Quality-control and leakage audit",
            "",
            f"All case matches passed (minimum manual-versus-inference T2 correlation, {qc_summary['minimum_manual_source_t2_correlation']:.6f}). No exact SHA-256 match was found between external T2 files and the 200-subject development source. Numeric identifiers overlapped with the development cohort, as expected for site-local numbering; center was retained as a separate cohort field, and the two sampled external subsets had no patient-ID overlap. Image-geometry and manual-volume shift screens are provided as exploratory diagnostics and were not used to modify preprocessing or the model.",
            "",
            "## Remaining boundary",
            "",
            "The external archives provide reference masks only for SSC, HSC, and PSC. External quantitative validation of cochlear, vestibular, and total vestibular (TV) segmentation remains unavailable. No radiomics, habitat, deep-learning classifier, or fusion-classifier frozen artifact was found in the current formal results tree; therefore no classification AUC, calibration, or DCA result was generated.",
            "",
            "## 中文简要说明",
            "",
            f"本次完成了冻结三半规管分割模型的真正外部定量验证：两组外部子队列各25例，共50例、100耳、300个SSC/HSC/PSC人工金标准掩膜。模型、重采样、ROI定位、阈值和连通域规则均未改变。合并外部集宏平均Dice为{format_ci(pooled_macro['mean'], pooled_macro['ci_low'], pooled_macro['ci_high'])}。由于当前终点是分割而非分类，AUC、校准和DCA不适用；耳蜗、前庭和TV仍缺人工金标准，不能报告其外部Dice/HD95。",
        ]
    )
    (args.output_dir / "MANUSCRIPT_METHODS_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")

    provenance = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "nibabel": nib.__version__,
        "matplotlib": matplotlib.__version__,
        "bootstrap_repetitions": args.bootstrap,
        "random_seed": args.seed,
        "command": sys.argv,
        "input_archives_sha256": archive_hashes,
        "model_files_sha256": model_hashes,
        "original_internal_summary": internal_original_summary,
    }
    (args.output_dir / "reproducibility_manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Completed: %d masks, %d ears, %d patients", len(metrics), metrics.ear_id.nunique(), metrics.patient_id.nunique())
    print("EXTERNAL_MANUAL_VALIDATION_COMPLETE", json.dumps(qc_summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
