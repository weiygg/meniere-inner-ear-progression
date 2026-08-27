from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
from scipy.ndimage import distance_transform_edt

from ..exceptions import ProtocolViolation


STRUCTURES = ("SSC", "HSC", "PSC")


def nnunet_case_id(subject_uid: str, ear_side: str) -> str:
    """Return a PHI-free nnU-Net case key for one ear crop."""
    side = ear_side.upper()
    if not subject_uid.startswith("LS_SEG_") or side not in {"L", "R"}:
        raise ProtocolViolation(f"Invalid LS segmentation case: {subject_uid}/{ear_side}")
    local = subject_uid.removeprefix("LS_SEG_")
    if not local.isdigit():
        raise ProtocolViolation(f"Invalid LS segmentation subject UID: {subject_uid}")
    return f"LSSEG{int(local):04d}{side}"


def overlap_audit(mask_stack: np.ndarray) -> dict[str, object]:
    masks = _validated_stack(mask_stack)
    count = masks.sum(axis=0)
    return {
        "overlap_voxels": int((count > 1).sum()),
        "triple_overlap_voxels": int((count == 3).sum()),
        "pair_overlap_voxels": {
            "SSC_HSC": int((masks[0] & masks[1]).sum()),
            "SSC_PSC": int((masks[0] & masks[2]).sum()),
            "HSC_PSC": int((masks[1] & masks[2]).sum()),
        },
    }


def combine_multiclass_masks(
    mask_stack: np.ndarray,
    *,
    overlap_policy: str = "fail",
    sampling_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, dict[str, object]]:
    """Convert three binary masks into one label map without silent priority.

    ``nearest-exclusive`` assigns each shared voxel to the structure whose
    non-overlapping core is closest in physical space. Exact ties use the
    frozen SSC, HSC, PSC order and are counted in the returned audit.
    """
    masks = _validated_stack(mask_stack)
    audit = overlap_audit(masks)
    overlap = masks.sum(axis=0) > 1
    if overlap.any() and overlap_policy == "fail":
        raise ProtocolViolation(
            f"Binary labels overlap in {audit['overlap_voxels']} voxels; "
            "an explicit conversion policy is required."
        )
    if overlap_policy not in {"fail", "nearest-exclusive"}:
        raise ProtocolViolation(f"Unsupported overlap policy: {overlap_policy}")

    labels = np.zeros(masks.shape[1:], dtype=np.uint8)
    exclusive = masks & (masks.sum(axis=0, keepdims=True) == 1)
    for index in range(3):
        labels[exclusive[index]] = index + 1

    assigned = {name: 0 for name in STRUCTURES}
    tie_count = 0
    if overlap.any():
        distances = []
        for index in range(3):
            if exclusive[index].any():
                distances.append(distance_transform_edt(~exclusive[index], sampling=sampling_mm))
            else:
                distances.append(np.full(labels.shape, np.inf, dtype=np.float64))
        distance_stack = np.stack(distances)
        candidates = np.where(masks, distance_stack, np.inf)
        chosen = np.argmin(candidates, axis=0)
        minimum = np.min(candidates, axis=0)
        tied_channels = (np.isclose(candidates, minimum[None]) & masks).sum(axis=0)
        tie_count = int((tied_channels[overlap] > 1).sum())
        labels[overlap] = chosen[overlap].astype(np.uint8) + 1
        for index, name in enumerate(STRUCTURES):
            assigned[name] = int((overlap & (chosen == index)).sum())

    audit["overlap_policy"] = overlap_policy
    audit["overlap_assignment_voxels"] = assigned
    audit["overlap_tie_voxels"] = tie_count
    return labels, audit


def locked_nnunet_split(rows: Iterable[Mapping[str, object]]) -> list[dict[str, list[str]]]:
    train: list[str] = []
    validation: list[str] = []
    test: list[str] = []
    for row in rows:
        case = nnunet_case_id(str(row["subject_uid"]), str(row["ear_side"]))
        split = str(row["split"])
        if split == "train":
            train.append(case)
        elif split == "validation":
            validation.append(case)
        elif split == "test":
            test.append(case)
        else:
            raise ProtocolViolation(f"Unexpected split: {split}")
    if len(train) != 280 or len(validation) != 60 or len(test) != 60:
        raise ProtocolViolation(
            f"Expected 280/60/60 ears, got {len(train)}/{len(validation)}/{len(test)}"
        )
    if set(train) & set(validation) or (set(train) | set(validation)) & set(test):
        raise ProtocolViolation("Case IDs cross locked split boundaries")
    return [{"train": sorted(train), "val": sorted(validation)}]


def _validated_stack(mask_stack: np.ndarray) -> np.ndarray:
    values = np.asarray(mask_stack)
    if values.ndim != 4 or values.shape[0] != 3:
        raise ProtocolViolation(f"Expected mask shape (3, X, Y, Z), got {values.shape}")
    unique = set(np.unique(values).tolist())
    if not unique.issubset({0, 1, False, True}):
        raise ProtocolViolation(f"Masks must be binary, got values {sorted(unique)}")
    return values.astype(bool, copy=False)
