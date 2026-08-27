from __future__ import annotations

from dataclasses import dataclass

from ..exceptions import ProtocolBlocker, ProtocolViolation


@dataclass(frozen=True)
class AffectedEarProtocol:
    exact_features_confirmed: bool
    abnormal_thresholds_confirmed: bool
    timing_window_confirmed: bool
    bilateral_rule_confirmed: bool
    index_ear_rule_confirmed: bool

    def assert_ready(self) -> None:
        missing = [
            name
            for name, value in vars(self).items()
            if not value
        ]
        if missing:
            raise ProtocolBlocker(f"Affected-ear protocol unresolved: {', '.join(missing)}")


def affected_ear_from_stage_presence(*_: object, **__: object) -> str:
    raise ProtocolViolation("AAO-HNS stage presence must not define the affected ear")


def default_to_numerically_worse_ear(*_: object, **__: object) -> str:
    raise ProtocolViolation("Numerically worse ear is not a default reference standard")
