from __future__ import annotations

from meniere_progression.pebm.eligibility import evaluate_variable


def row(variable: str, role: str) -> dict[str, object]:
    return {
        "variable": variable,
        "source_center": "both",
        "unit": "documented",
        "coding": "documented",
        "missing_code": "documented",
        "abnormal_direction": "higher_is_abnormal",
        "measurement_window": "documented",
        "variable_type": "continuous",
        "supported_mixture": True,
        "cross_center_compatible": True,
        "collinearity_group": "none",
        "candidate_role": role,
    }


def test_nonmonotonic_validators_not_primary_events() -> None:
    result = evaluate_variable(row("DHI", "primary_event"))
    assert result["eligibility_status"] == "blocked"
    assert "nonmonotonic_validator" in result["blocking_reason"]


def test_static_geometry_not_primary_event_by_default() -> None:
    result = evaluate_variable(row("SSC_geometry", "primary_event"))
    assert result["eligibility_status"] == "blocked"
    assert "static_geometry" in result["blocking_reason"]
