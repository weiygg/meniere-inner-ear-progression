from __future__ import annotations

import re
from collections.abc import Iterable

from .exceptions import ProtocolViolation


DATASET_PREFIXES = {
    "LS_SEG_200": "LS_SEG",
    "LS_CLIN_79": "LS_CLIN",
    "Z2_SEG_EXT1": "Z2_SEG_EXT1",
    "Z2_SEG_EXT2": "Z2_SEG_EXT2",
    "EXT_MANUAL_50": "Z2_SEG_MANUAL",
    "Z2_CLIN": "Z2_CLIN",
}
_BARE_NUMERIC = re.compile(r"^\d+$")
_SAFE_SOURCE_ID = re.compile(r"[^A-Za-z0-9_-]+")


def normalize_source_id(source_id: object, width: int = 4) -> str:
    """Normalize a dataset-local ID without treating it as a global identity."""
    value = str(source_id).strip()
    if not value:
        raise ValueError("source_id must not be empty")
    if _BARE_NUMERIC.fullmatch(value):
        return value.zfill(width)
    cleaned = _SAFE_SOURCE_ID.sub("_", value).strip("_").upper()
    if not cleaned:
        raise ValueError("source_id has no usable characters")
    return cleaned


def subject_uid(dataset_id: str, source_id: object) -> str:
    try:
        prefix = DATASET_PREFIXES[dataset_id]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset_id: {dataset_id}") from exc
    return f"{prefix}_{normalize_source_id(source_id)}"


def visit_uid(subject: str, visit_id: object) -> str:
    assert_namespaced_uid(subject)
    return f"{subject}__VISIT_{normalize_source_id(visit_id, width=3)}"


def ear_uid(subject: str, side: str) -> str:
    assert_namespaced_uid(subject)
    side = side.upper().strip()
    if side not in {"L", "R"}:
        raise ValueError("side must be L or R")
    return f"{subject}__EAR_{side}"


def assert_namespaced_uid(value: str) -> None:
    if _BARE_NUMERIC.fullmatch(str(value).strip()):
        raise ProtocolViolation(f"Bare numeric ID is forbidden: {value!r}")
    if not any(str(value).startswith(f"{prefix}_") for prefix in DATASET_PREFIXES.values()):
        raise ProtocolViolation(f"ID lacks an approved dataset namespace: {value!r}")


def assert_unique_namespaced_ids(values: Iterable[str]) -> None:
    values = list(values)
    for value in values:
        assert_namespaced_uid(value)
    if len(values) != len(set(values)):
        raise ProtocolViolation("Namespaced IDs are not unique")


def assert_no_bare_numeric_cross_dataset_join(
    left_dataset: str,
    right_dataset: str,
    left_keys: Iterable[object],
    right_keys: Iterable[object],
) -> None:
    """Fail when different datasets are joined using only local numeric keys."""
    if left_dataset == right_dataset:
        return
    combined = [str(value).strip() for value in (*left_keys, *right_keys)]
    if combined and all(_BARE_NUMERIC.fullmatch(value) for value in combined):
        raise ProtocolViolation(
            f"Bare numeric cross-dataset join forbidden: {left_dataset} <-> {right_dataset}"
        )
