from __future__ import annotations

import numpy as np
import pytest

from meniere_progression.exceptions import ProtocolViolation
from meniere_progression.segmentation.multiclass import (
    combine_multiclass_masks,
    locked_nnunet_split,
    nnunet_case_id,
)


def test_overlap_must_not_be_silently_overwritten() -> None:
    masks = np.zeros((3, 5, 5, 5), dtype=np.uint8)
    masks[0, 1:4, 2, 2] = 1
    masks[2, 2:5, 2, 2] = 1
    with pytest.raises(ProtocolViolation):
        combine_multiclass_masks(masks)
    labels, audit = combine_multiclass_masks(masks, overlap_policy="nearest-exclusive")
    assert set(np.unique(labels)) == {0, 1, 3}
    assert audit["overlap_voxels"] == 2
    assert sum(audit["overlap_assignment_voxels"].values()) == 2


def test_case_ids_are_namespaced_and_side_specific() -> None:
    assert nnunet_case_id("LS_SEG_0007", "L") == "LSSEG0007L"
    with pytest.raises(ProtocolViolation):
        nnunet_case_id("LS_CLIN_0007", "L")


def test_locked_split_requires_protocol_counts() -> None:
    rows = []
    for subject in range(1, 201):
        split = "train" if subject <= 140 else "validation" if subject <= 170 else "test"
        for side in ("L", "R"):
            rows.append(
                {"subject_uid": f"LS_SEG_{subject:04d}", "ear_side": side, "split": split}
            )
    split = locked_nnunet_split(rows)[0]
    assert len(split["train"]) == 280
    assert len(split["val"]) == 60
