from __future__ import annotations

import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("nnunetv2") is None or importlib.util.find_spec("monai") is None,
    reason="optional segmentation training dependencies are absent",
)


def test_protocol_trainers_import_and_prespecification() -> None:
    from meniere_progression.segmentation.nnunet_trainers import (
        BoundedBiasFieldTransform,
        nnUNetTrainerProtocolV2M2,
        nnUNetTrainerProtocolV2M3,
    )

    assert BoundedBiasFieldTransform().coefficient_range == (-0.15, 0.15)
    assert issubclass(nnUNetTrainerProtocolV2M3, nnUNetTrainerProtocolV2M2)
