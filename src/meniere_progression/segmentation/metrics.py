from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def _binary(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=bool)


def dice(prediction: np.ndarray, reference: np.ndarray) -> float:
    prediction = _binary(prediction)
    reference = _binary(reference)
    intersection = int(np.logical_and(prediction, reference).sum())
    denominator = int(prediction.sum()) + int(reference.sum())
    return (2.0 * intersection + 1e-5) / (denominator + 1e-5)


def precision_recall(prediction: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    prediction = _binary(prediction)
    reference = _binary(reference)
    intersection = int(np.logical_and(prediction, reference).sum())
    precision = (intersection + 1e-5) / (int(prediction.sum()) + 1e-5)
    recall = (intersection + 1e-5) / (int(reference.sum()) + 1e-5)
    return precision, recall


def soft_cldice(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Hard-mask clDice audit metric using 3D skeletons."""
    prediction = _binary(prediction)
    reference = _binary(reference)
    if not prediction.any() and not reference.any():
        return 1.0
    skeleton_prediction = skeletonize(prediction)
    skeleton_reference = skeletonize(reference)
    topology_precision = (
        np.logical_and(skeleton_prediction, reference).sum() + 1e-5
    ) / (skeleton_prediction.sum() + 1e-5)
    topology_sensitivity = (
        np.logical_and(skeleton_reference, prediction).sum() + 1e-5
    ) / (skeleton_reference.sum() + 1e-5)
    return float(
        2.0 * topology_precision * topology_sensitivity
        / (topology_precision + topology_sensitivity + 1e-5)
    )


def surface_distances(
    prediction: np.ndarray, reference: np.ndarray, spacing: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    prediction = _binary(prediction)
    reference = _binary(reference)
    structure = ndimage.generate_binary_structure(3, 1)
    prediction_surface = prediction ^ ndimage.binary_erosion(prediction, structure=structure)
    reference_surface = reference ^ ndimage.binary_erosion(reference, structure=structure)
    if not prediction_surface.any() or not reference_surface.any():
        return np.asarray([np.inf]), np.asarray([np.inf])
    to_reference = ndimage.distance_transform_edt(~reference_surface, sampling=spacing)
    to_prediction = ndimage.distance_transform_edt(~prediction_surface, sampling=spacing)
    return to_reference[prediction_surface], to_prediction[reference_surface]


def surface_summary(
    prediction: np.ndarray,
    reference: np.ndarray,
    spacing: tuple[float, float, float],
    tolerance_mm: float = 1.0,
) -> dict[str, float]:
    pred_to_ref, ref_to_pred = surface_distances(prediction, reference, spacing)
    combined = np.concatenate([pred_to_ref, ref_to_pred])
    return {
        "ASSD_mm": float(combined.mean()),
        "HD95_mm": float(np.percentile(combined, 95)),
        "surface_dice_1mm": float(
            (np.count_nonzero(pred_to_ref <= tolerance_mm) + np.count_nonzero(ref_to_pred <= tolerance_mm))
            / (len(pred_to_ref) + len(ref_to_pred))
        ),
    }
