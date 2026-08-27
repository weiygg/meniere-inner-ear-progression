from __future__ import annotations

import pytest

from meniere_progression.exceptions import ProtocolViolation
from meniere_progression.identifiers import (
    assert_no_bare_numeric_cross_dataset_join,
    assert_unique_namespaced_ids,
    ear_uid,
    subject_uid,
)


def test_namespaced_ids_are_unique() -> None:
    values = [subject_uid("LS_SEG_200", 1), subject_uid("LS_CLIN_79", 1)]
    assert values == ["LS_SEG_0001", "LS_CLIN_0001"]
    assert_unique_namespaced_ids(values)
    assert ear_uid(values[0], "R") == "LS_SEG_0001__EAR_R"


def test_no_bare_numeric_cross_dataset_join() -> None:
    with pytest.raises(ProtocolViolation):
        assert_no_bare_numeric_cross_dataset_join(
            "LS_SEG_200", "LS_CLIN_79", [1, 2, 3], [1, 2, 3]
        )


def test_ls_seg_vs_ls_clin_same_local_number_is_not_same_uid() -> None:
    assert subject_uid("LS_SEG_200", 79) != subject_uid("LS_CLIN_79", 79)
