from __future__ import annotations

import pytest

from conftest import load_script


summary_module = load_script("../scripts/summarize_nnunet_validation.py")


def test_patient_clustered_summary_preserves_both_ears() -> None:
    cases = []
    for patient, base in (("0001", 0.7), ("0002", 0.9)):
        for side in ("L", "R"):
            cases.append(
                {
                    "prediction_file": f"LSSEG{patient}{side}.nii.gz",
                    "metrics": {
                        "1": {"Dice": base},
                        "2": {"Dice": base + 0.05},
                        "3": {"Dice": base - 0.05},
                    },
                }
            )
    result = summary_module.summarize({"metric_per_case": cases}, repetitions=100, seed=7)
    assert result["people"] == 2
    assert result["ears"] == 4
    assert result["estimates"]["macro"]["dice"] == pytest.approx(0.8)
