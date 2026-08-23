from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inner_ear_vit_seg_experiment import TinyViTUNet3D, crop_with_padding, normalize_intensity, resample_volume


ADDITIONAL_STRUCTURES = ("Cochlear", "Vestibular", "TV")
CANAL_STRUCTURES = ("SSC", "HSC", "PSC")
ALL_T2_STRUCTURES = ("Cochlear", "Vestibular", "SSC", "HSC", "PSC", "TV")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer additional T2 masks and split all six structures by external center.")
    parser.add_argument(
        "--nifti-dir",
        type=Path,
        default=Path("results_md_progression/intermediate/semicircular_canal_vit_20260731/z2_prepared/nifti"),
    )
    parser.add_argument(
        "--center-linkage",
        type=Path,
        default=Path("results_md_progression/final/study_design_corrected_20260801/audit/external_center_clinical_linkage.csv"),
    )
    parser.add_argument(
        "--frozen-center-qc",
        type=Path,
        default=Path(
            "results_md_progression/final/semicircular_canal_vit_20260731/"
            "external_inference_v2_ensemble/external_study_qc.csv"
        ),
    )
    parser.add_argument(
        "--frozen-canal-qc",
        type=Path,
        default=Path(
            "results_md_progression/final/semicircular_canal_vit_20260731/"
            "external_inference_v2_ensemble/external_mask_qc.csv"
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("results_md_progression/final/all_t2_inner_ear_vit_20260801/additional_models"),
    )
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=Path("results_md_progression/intermediate/all_t2_vit_20260801/additional_training_crops/sample_manifest.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results_md_progression/final/all_t2_inner_ear_vit_20260801"),
    )
    parser.add_argument("--centers", default="center2,center3")
    parser.add_argument("--crop-size", nargs=3, type=int, default=(128, 128, 48))
    parser.add_argument("--target-spacing", nargs=3, type=float, default=(0.3472222, 0.3472222, 0.5))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_center(value: str) -> np.ndarray:
    return np.asarray([float(item) for item in value.split(",")], dtype=np.float32)


def crop_bounds(full_shape: tuple[int, int, int], center: np.ndarray, crop_size: tuple[int, int, int]):
    full_slices = []
    crop_slices = []
    for dim, size in enumerate(crop_size):
        raw_start = int(round(float(center[dim]) - size / 2))
        raw_end = raw_start + size
        full_start = max(0, raw_start)
        full_end = min(full_shape[dim], raw_end)
        crop_start = full_start - raw_start
        crop_end = crop_start + (full_end - full_start)
        full_slices.append(slice(full_start, full_end))
        crop_slices.append(slice(crop_start, crop_end))
    return tuple(full_slices), tuple(crop_slices)


def retain_components(mask: np.ndarray, top_k: int) -> tuple[np.ndarray, int]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if top_k == 0 or count <= top_k:
        return mask.astype(np.uint8), count
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    selected = np.argsort(sizes)[-top_k:]
    return np.isin(labels, selected).astype(np.uint8), count


def boundary_touch(mask: np.ndarray) -> bool:
    return bool(
        mask.any()
        and (
            mask[0].any()
            or mask[-1].any()
            or mask[:, 0].any()
            or mask[:, -1].any()
            or mask[:, :, 0].any()
            or mask[:, :, -1].any()
        )
    )


def training_volume_reference(rows: list[dict[str, str]], voxel_volume: float) -> dict[str, dict[str, float]]:
    reference: dict[str, dict[str, float]] = {}
    for structure in ADDITIONAL_STRUCTURES:
        volumes = np.asarray(
            [float(row["full_voxels"]) * voxel_volume for row in rows if row["structure"] == structure and row["split"] == "train"],
            dtype=float,
        )
        reference[structure] = {
            "p01": float(np.percentile(volumes, 1)),
            "median": float(np.median(volumes)),
            "p99": float(np.percentile(volumes, 99)),
        }
    return reference


def main() -> None:
    args = parse_args()
    requested_centers = {item.strip() for item in args.centers.split(",") if item.strip()}
    crop_size = tuple(args.crop_size)
    target_spacing = tuple(args.target_spacing)
    voxel_volume = float(np.prod(target_spacing))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    linkage = read_csv(args.center_linkage)
    center_by_study = {row["study_id"]: row["center"] for row in linkage if row["center"] in requested_centers}
    frozen_centers = {row["study_id"]: row for row in read_csv(args.frozen_center_qc)}
    frozen_canal_rows = read_csv(args.frozen_canal_qc)
    canal_by_center: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in frozen_canal_rows:
        center = center_by_study.get(row["study_id"])
        if center:
            canal_by_center[center].append({"center": center, "inference_source": "frozen_20260731_canal_model", **row})

    metrics_summary = json.loads((args.model_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    policies = {
        structure: metrics_summary["structures"][structure]["selected_postprocessing"]
        for structure in ADDITIONAL_STRUCTURES
    }
    models: dict[str, torch.nn.Module] = {}
    for structure in ADDITIONAL_STRUCTURES:
        checkpoint = torch.load(args.model_dir / f"best_{structure}_model.pt", map_location=device, weights_only=False)
        model = TinyViTUNet3D(crop_size).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models[structure] = model

    volume_reference = training_volume_reference(read_csv(args.training_manifest), voxel_volume)
    additional_by_center: dict[str, list[dict[str, object]]] = defaultdict(list)
    study_status_by_center: dict[str, list[dict[str, object]]] = defaultdict(list)
    selected_studies = sorted(center_by_study)
    for study_index, study_id in enumerate(selected_studies, start=1):
        center = center_by_study[study_id]
        center_row = frozen_centers.get(study_id)
        input_path = args.nifti_dir / f"{study_id}_T2.nii.gz"
        if center_row is None or not input_path.exists():
            study_status_by_center[center].append(
                {"center": center, "study_id": study_id, "qc_status": "error", "qc_flags": "missing_frozen_center_or_nifti"}
            )
            continue
        canonical = nib.as_closest_canonical(nib.load(str(input_path)))
        image_raw = np.asarray(canonical.dataobj, dtype=np.float32)
        spacing = tuple(float(value) for value in canonical.header.get_zooms()[:3])
        image_resampled = normalize_intensity(resample_volume(image_raw, spacing, target_spacing, order=1))
        new_affine = nib.affines.rescale_affine(canonical.affine, canonical.shape[:3], target_spacing, image_resampled.shape)
        centers = {
            "L": parse_center(center_row["left_center_voxels"]),
            "R": parse_center(center_row["right_center_voxels"]),
        }
        crops = np.stack([crop_with_padding(image_resampled, centers[side], crop_size) for side in ("L", "R")]).astype(np.float32)
        image_tensor = torch.from_numpy(crops[:, None]).to(device)
        probabilities: dict[str, np.ndarray] = {}
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            for structure in ADDITIONAL_STRUCTURES:
                probabilities[structure] = torch.sigmoid(models[structure](image_tensor)).float().cpu().numpy()[:, 0]

        study_warnings: list[str] = []
        study_dir = args.output_root / f"external_{center}" / "predicted_masks" / f"sub{study_id}"
        study_dir.mkdir(parents=True, exist_ok=True)
        for side_index, side in enumerate(("L", "R")):
            full_slices, crop_slices = crop_bounds(image_resampled.shape, centers[side], crop_size)
            for structure in ADDITIONAL_STRUCTURES:
                policy = policies[structure]
                probability = probabilities[structure][side_index]
                processed, component_count = retain_components(
                    probability > float(policy["threshold"]), int(policy["top_k_components"])
                )
                full_mask = np.zeros(image_resampled.shape, dtype=np.uint8)
                full_mask[full_slices] = processed[crop_slices]
                voxel_count = int(full_mask.sum())
                volume = voxel_count * voxel_volume
                flags: list[str] = []
                if voxel_count == 0:
                    flags.append("empty_prediction")
                elif voxel_count < 20:
                    flags.append("tiny_prediction")
                if boundary_touch(processed):
                    flags.append("touches_crop_boundary")
                if component_count > 3:
                    flags.append("fragmented_before_component_filter")
                reference = volume_reference[structure]
                volume_status = "pass" if reference["p01"] <= volume <= reference["p99"] else "warning"
                if volume_status == "warning":
                    flags.append("outside_LS_training_p01_p99_volume")
                output_path = study_dir / f"{study_id}{side}_{structure}.nii.gz"
                nib.save(nib.Nifti1Image(full_mask, new_affine), str(output_path))
                additional_by_center[center].append(
                    {
                        "center": center,
                        "study_id": study_id,
                        "ear_side": side,
                        "structure": structure,
                        "inference_source": "frozen_20260801_additional_structure_model",
                        "threshold": policy["threshold"],
                        "top_k_components": policy["top_k_components"],
                        "predicted_voxels": voxel_count,
                        "predicted_volume_mm3": volume,
                        "mean_foreground_probability": float(probability[processed > 0].mean()) if processed.any() else 0.0,
                        "components_before_filter": component_count,
                        "touches_crop_boundary": boundary_touch(processed),
                        "LS_training_volume_p01_mm3": reference["p01"],
                        "LS_training_volume_median_mm3": reference["median"],
                        "LS_training_volume_p99_mm3": reference["p99"],
                        "volume_plausibility_status": volume_status,
                        "qc_status": "warning" if flags else "pass",
                        "qc_flags": ";".join(flags),
                        "mask_path": str(output_path.resolve()),
                        "analysis_eligible_without_manual_review": False,
                    }
                )
                study_warnings.extend(flags)
        study_status_by_center[center].append(
            {
                "center": center,
                "study_id": study_id,
                "input_nifti": str(input_path.resolve()),
                "qc_status": "warning" if study_warnings else "pass",
                "qc_flags": ";".join(sorted(set(study_warnings))),
                "analysis_eligible_without_manual_review": False,
            }
        )
        print(f"EXTERNAL_INFERENCE {study_index}/{len(selected_studies)} {center} {study_id}", flush=True)

    combined_summary: dict[str, object] = {"device": str(device), "centers": {}}
    for center in sorted(requested_centers):
        output_dir = args.output_root / f"external_{center}"
        additional_rows = additional_by_center[center]
        canal_rows = canal_by_center[center]
        combined_rows = sorted(
            [
                {
                    "center": row["center"],
                    "study_id": row["study_id"],
                    "ear_side": row["ear_side"],
                    "structure": row["structure"],
                    "inference_source": row["inference_source"],
                    "predicted_volume_mm3": row["predicted_volume_mm3"],
                    "qc_status": row["qc_status"],
                    "qc_flags": row["qc_flags"],
                    "mask_path": row["mask_path"],
                    "analysis_eligible_without_manual_review": False,
                }
                for row in [*canal_rows, *additional_rows]
            ],
            key=lambda row: (str(row["study_id"]), str(row["ear_side"]), ALL_T2_STRUCTURES.index(str(row["structure"]))),
        )
        write_csv(output_dir / "additional_mask_qc.csv", additional_rows)
        write_csv(output_dir / "all_six_structure_mask_manifest.csv", combined_rows)
        write_csv(output_dir / "study_qc.csv", study_status_by_center[center])
        studies = {str(row["study_id"]) for row in combined_rows}
        expected_masks = len(studies) * 2 * len(ALL_T2_STRUCTURES)
        center_summary = {
            "study_count": len(studies),
            "ear_count": len(studies) * 2,
            "structures": list(ALL_T2_STRUCTURES),
            "combined_mask_rows": len(combined_rows),
            "expected_mask_rows": expected_masks,
            "complete_six_structure_inference": len(combined_rows) == expected_masks,
            "additional_mask_qc_counts": dict(Counter(row["qc_status"] for row in additional_rows)),
            "additional_volume_qc_counts": dict(Counter(row["volume_plausibility_status"] for row in additional_rows)),
            "external_validation_boundary": (
                "Automatic mask inference and technical QC only. External Dice/HD95 require manual reference masks; "
                "downstream feature analysis requires manual mask review."
            ),
        }
        (output_dir / "external_inference_summary.json").write_text(
            json.dumps(center_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        combined_summary["centers"][center] = center_summary
    (args.output_root / "external_centers_summary.json").write_text(
        json.dumps(combined_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("EXTERNAL_CENTER_INFERENCE_COMPLETE", json.dumps(combined_summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
