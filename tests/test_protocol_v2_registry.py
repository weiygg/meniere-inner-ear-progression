from __future__ import annotations

from pathlib import Path

import pytest

from meniere_progression.exceptions import ProtocolViolation
from meniere_progression.registry import DatasetRegistry, validate_overlap_file


ROOT = Path(__file__).resolve().parents[1]


def test_ls_seg_vs_ls_clin_overlap_zero() -> None:
    validate_overlap_file(ROOT / "configs/dataset_overlap.yaml")


def test_external_never_enters_training() -> None:
    registry = DatasetRegistry.load(ROOT / "configs/dataset_registry.yaml")
    registry.validate()
    for dataset_id in ("Z2_SEG_EXT1", "Z2_SEG_EXT2", "EXT_MANUAL_50"):
        with pytest.raises(ProtocolViolation):
            registry.assert_allowed(dataset_id, "training")


def test_external_never_enters_threshold_selection() -> None:
    registry = DatasetRegistry.load(ROOT / "configs/dataset_registry.yaml")
    for dataset_id in ("Z2_SEG_EXT1", "Z2_SEG_EXT2", "EXT_MANUAL_50"):
        with pytest.raises(ProtocolViolation):
            registry.assert_allowed(dataset_id, "threshold_selection")
    registry.assert_allowed("LS_SEG_200", "threshold_selection")
