from __future__ import annotations

from collections.abc import Iterable

from ..exceptions import ProtocolViolation


PROHIBITED_AUTOMATIC_JOIN_FIELDS = {"name", "姓名", "row", "row_number", "approximate_date"}
REQUIRED_CROSSWALK_FIELDS = {"source_subject_uid", "canonical_subject_uid", "documented_visit_rule"}


def validate_crosswalk_fields(fields: Iterable[str]) -> None:
    fields = set(fields)
    prohibited = fields & PROHIBITED_AUTOMATIC_JOIN_FIELDS
    if prohibited:
        raise ProtocolViolation(f"Prohibited automatic crosswalk fields: {sorted(prohibited)}")
    missing = REQUIRED_CROSSWALK_FIELDS - fields
    if missing:
        raise ProtocolViolation(f"Explicit crosswalk fields missing: {sorted(missing)}")
