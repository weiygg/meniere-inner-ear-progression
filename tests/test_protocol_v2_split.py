from __future__ import annotations

import pytest

from meniere_progression.exceptions import ProtocolViolation
from meniere_progression.segmentation.datasets import validate_patient_level_split


def test_patient_level_split_keeps_both_ears_together() -> None:
    rows = [
        {"subject_uid": "LS_SEG_0001", "ear_side": "L", "split": "train"},
        {"subject_uid": "LS_SEG_0001", "ear_side": "R", "split": "train"},
        {"subject_uid": "LS_SEG_0002", "ear_side": "L", "split": "validation"},
        {"subject_uid": "LS_SEG_0002", "ear_side": "R", "split": "validation"},
    ]
    validate_patient_level_split(rows)
    rows[-1]["split"] = "test"
    with pytest.raises(ProtocolViolation):
        validate_patient_level_split(rows)
