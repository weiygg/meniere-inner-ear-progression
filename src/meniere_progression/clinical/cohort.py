from __future__ import annotations

from collections.abc import Iterable

from ..identifiers import assert_unique_namespaced_ids


def validate_subject_master(subject_uids: Iterable[str]) -> None:
    assert_unique_namespaced_ids(subject_uids)
