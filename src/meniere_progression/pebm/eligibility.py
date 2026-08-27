from __future__ import annotations

from collections.abc import Mapping


REQUIRED_FIELDS = {
    "variable",
    "source_center",
    "unit",
    "coding",
    "missing_code",
    "abnormal_direction",
    "measurement_window",
    "variable_type",
    "supported_mixture",
    "cross_center_compatible",
    "collinearity_group",
    "candidate_role",
}
NONMONOTONIC_VALIDATORS = {"AAO_HNS_stage", "DHI", "THI", "VADL", "ear_fullness", "symptom_burden"}
STATIC_GEOMETRY = {"SSC_geometry", "HSC_geometry", "PSC_geometry", "VA", "ES_ED"}


def evaluate_variable(row: Mapping[str, object]) -> dict[str, object]:
    missing = sorted(REQUIRED_FIELDS - set(row))
    blockers: list[str] = []
    if missing:
        blockers.append(f"missing_fields:{','.join(missing)}")
    variable = str(row.get("variable", ""))
    role = str(row.get("candidate_role", ""))
    if variable in NONMONOTONIC_VALIDATORS and role == "primary_event":
        blockers.append("nonmonotonic_validator_not_primary_event")
    if variable in STATIC_GEOMETRY and role == "primary_event":
        blockers.append("static_geometry_not_primary_event_by_default")
    if not bool(row.get("supported_mixture", False)):
        blockers.append("unsupported_mixture")
    if not bool(row.get("cross_center_compatible", False)):
        blockers.append("cross_center_definition_or_window_mismatch")
    result = dict(row)
    result["eligibility_status"] = "eligible" if not blockers else "blocked"
    result["blocking_reason"] = ";".join(blockers)
    return result
