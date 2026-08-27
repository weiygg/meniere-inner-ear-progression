from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

from ..exceptions import ProtocolViolation
from ..identifiers import assert_namespaced_uid


def validate_patient_level_split(rows: Iterable[Mapping[str, object]]) -> None:
    """Ensure every subject, including both ears, belongs to exactly one split."""
    splits: dict[str, set[str]] = defaultdict(set)
    sides: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        subject = str(row["subject_uid"])
        split = str(row["split"])
        side = str(row.get("ear_side", "")).upper()
        assert_namespaced_uid(subject)
        splits[subject].add(split)
        if side:
            if side not in {"L", "R"}:
                raise ProtocolViolation(f"Invalid ear side: {side}")
            sides[subject].add(side)
    leaked = sorted(subject for subject, values in splits.items() if len(values) != 1)
    if leaked:
        raise ProtocolViolation(f"Subjects cross split boundaries: {leaked[:10]}")
    incomplete = sorted(subject for subject, values in sides.items() if values != {"L", "R"})
    if incomplete:
        raise ProtocolViolation(f"Subjects do not have both ears in the manifest: {incomplete[:10]}")


def aggregate_split_counts(rows: Iterable[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    rows = list(rows)
    result: dict[str, dict[str, int]] = {}
    for split in sorted({str(row["split"]) for row in rows}):
        block = [row for row in rows if str(row["split"]) == split]
        result[split] = {
            "people": len({str(row["subject_uid"]) for row in block}),
            "ears": len(block),
        }
    return result
