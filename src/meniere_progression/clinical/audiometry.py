from __future__ import annotations

from collections.abc import Mapping

from ..exceptions import ProtocolBlocker, ProtocolViolation


PTA_FREQUENCIES_KHZ = (0.5, 1.0, 2.0, 3.0)


def pta_aaohns_05123(
    thresholds_db: Mapping[float, float], *, unit_confirmed_db_hl: bool = True
) -> float:
    """Recompute the 0.5/1/2/3-kHz arithmetic mean from source thresholds."""
    if not unit_confirmed_db_hl:
        raise ProtocolBlocker("Audiometry unit dB HL has not been confirmed")
    missing = [frequency for frequency in PTA_FREQUENCIES_KHZ if frequency not in thresholds_db]
    if missing:
        raise ValueError(f"Missing PTA frequencies: {missing}")
    return sum(float(thresholds_db[frequency]) for frequency in PTA_FREQUENCIES_KHZ) / 4.0


def reject_fake_4khz_pta(thresholds_db: Mapping[float, float]) -> None:
    if 4.0 not in thresholds_db:
        raise ProtocolViolation("A 4-kHz PTA cannot be constructed without source 4-kHz data")
