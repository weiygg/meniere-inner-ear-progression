from __future__ import annotations

import pytest

from meniere_progression.clinical.affected_ear import affected_ear_from_stage_presence
from meniere_progression.clinical.audiometry import pta_aaohns_05123, reject_fake_4khz_pta
from meniere_progression.exceptions import ProtocolViolation


def test_pta_uses_05_1_2_3khz() -> None:
    assert pta_aaohns_05123({0.5: 10, 1.0: 20, 2.0: 30, 3.0: 40}) == 25


def test_no_fake_4khz_pta() -> None:
    with pytest.raises(ProtocolViolation):
        reject_fake_4khz_pta({0.5: 10, 1.0: 20, 2.0: 30, 3.0: 40})


def test_affected_ear_not_from_stage_presence() -> None:
    with pytest.raises(ProtocolViolation):
        affected_ear_from_stage_presence(True, False)
