from __future__ import annotations

import numpy as np

from meniere_progression.segmentation.reliability import paired_reliability


def test_perfect_geometry_pairs_have_icc_one_and_zero_bias() -> None:
    values = np.asarray([1.0, 2.0, 4.0, 8.0])
    estimate = paired_reliability(values, values.copy())
    assert np.isclose(estimate.icc_a1, 1.0)
    assert np.isclose(estimate.bland_altman_bias, 0.0)
