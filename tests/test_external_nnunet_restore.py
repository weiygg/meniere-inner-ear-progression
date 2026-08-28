import numpy as np

from scripts.restore_nnunet_external_predictions import crop_bounds


def test_crop_bounds_without_padding() -> None:
    full, crop = crop_bounds((200, 200, 80), np.asarray([100, 100, 40]), (128, 128, 48))
    assert full == (slice(36, 164), slice(36, 164), slice(16, 64))
    assert crop == (slice(0, 128), slice(0, 128), slice(0, 48))


def test_crop_bounds_with_padding() -> None:
    full, crop = crop_bounds((100, 100, 40), np.asarray([10, 8, 5]), (128, 128, 48))
    assert full == (slice(0, 74), slice(0, 72), slice(0, 29))
    assert crop == (slice(54, 128), slice(56, 128), slice(19, 48))
