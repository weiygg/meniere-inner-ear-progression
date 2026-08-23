import numpy as np

from conftest import load_script


def test_centerline_geometry_returns_physical_units_for_tube():
    mod = load_script("05_extract_inner_ear_morphometry.py")
    mask = np.zeros((30, 11, 11), dtype=np.uint8)
    mask[3:27, 4:7, 4:7] = 1
    spacing = np.array([0.5, 0.8, 1.2])
    affine = np.diag([*spacing, 1.0])
    result = mod.centerline_features(mask, spacing, affine)
    assert 10.0 < result["centerline_length_mm"] < 13.0
    assert result["minimum_diameter_mm"] > 0
    assert result["plane_rms_residual_mm"] < 0.2


def test_plane_angle_is_orientation_sign_invariant():
    mod = load_script("05_extract_inner_ear_morphometry.py")
    assert np.isclose(mod._angle_degrees(np.array([1, 0, 0]), np.array([-1, 0, 0])), 0)
    assert np.isclose(mod._angle_degrees(np.array([1, 0, 0]), np.array([0, 1, 0])), 90)
