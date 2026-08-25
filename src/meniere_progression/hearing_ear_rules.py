from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import sqrt
from typing import Iterable, Mapping, Sequence


FREQUENCIES = ("0.5kHZ", "1kHZ", "2kHZ", "3kHZ")
THRESHOLDS_DB = (20.0, 25.0, 30.0, 35.0, 40.0)
RESOLUTION_MODES = (
    "strict",
    "pta_gap_10",
    "pta_gap_15",
    "pta_gap_20",
    "worse_pta_no_minimum",
)


@dataclass(frozen=True)
class EarAudiometry:
    side: str
    thresholds_db: tuple[float, float, float, float]
    stage_present: bool = False

    @property
    def pta_05123(self) -> float:
        return sum(self.thresholds_db) / 4.0

    @property
    def low_frequency_mean(self) -> float:
        return sum(self.thresholds_db[:2]) / 2.0


@dataclass(frozen=True)
class AbnormalityRule:
    rule_id: str
    family: str
    threshold_db: float
    minimum_frequency_count: int | None
    label: str

    def abnormal(self, ear: EarAudiometry) -> bool:
        if self.family == "pta_05123":
            return ear.pta_05123 > self.threshold_db
        if self.family == "frequency_count":
            assert self.minimum_frequency_count is not None
            return (
                sum(value > self.threshold_db for value in ear.thresholds_db)
                >= self.minimum_frequency_count
            )
        if self.family == "low_frequency_mean":
            return ear.low_frequency_mean > self.threshold_db
        raise ValueError(f"Unknown rule family: {self.family}")


def build_rule_grid() -> tuple[AbnormalityRule, ...]:
    rules: list[AbnormalityRule] = []
    for threshold in THRESHOLDS_DB:
        token = int(threshold)
        rules.append(
            AbnormalityRule(
                f"pta_05123_gt_{token}",
                "pta_05123",
                threshold,
                None,
                f"mean(0.5/1/2/3 kHz) > {token} dB HL",
            )
        )
    for threshold in THRESHOLDS_DB:
        token = int(threshold)
        for minimum_count in range(1, 5):
            rules.append(
                AbnormalityRule(
                    f"freq_gt_{token}_n{minimum_count}",
                    "frequency_count",
                    threshold,
                    minimum_count,
                    f"> {token} dB HL at >= {minimum_count} of 4 frequencies",
                )
            )
    for threshold in THRESHOLDS_DB:
        token = int(threshold)
        rules.append(
            AbnormalityRule(
                f"lowfreq_mean_gt_{token}",
                "low_frequency_mean",
                threshold,
                None,
                f"mean(0.5/1 kHz) > {token} dB HL",
            )
        )
    return tuple(rules)


def resolve_index_ear(
    left: EarAudiometry,
    right: EarAudiometry,
    rule: AbnormalityRule,
    mode: str,
) -> tuple[str, str | None]:
    """Return base abnormality state and an index side, if uniquely resolved."""
    if mode not in RESOLUTION_MODES:
        raise ValueError(f"Unknown resolution mode: {mode}")

    left_abnormal = rule.abnormal(left)
    right_abnormal = rule.abnormal(right)
    if left_abnormal and not right_abnormal:
        return "unilateral_abnormal", "L"
    if right_abnormal and not left_abnormal:
        return "unilateral_abnormal", "R"
    if not left_abnormal and not right_abnormal:
        return "neither_abnormal", None

    if mode == "strict":
        return "bilateral_abnormal", None

    pta_difference = abs(left.pta_05123 - right.pta_05123)
    if mode == "worse_pta_no_minimum":
        required_difference = 0.0
    else:
        required_difference = float(mode.rsplit("_", 1)[1])

    if pta_difference < required_difference or pta_difference == 0:
        return "bilateral_abnormal", None
    return (
        "bilateral_abnormal",
        "L" if left.pta_05123 > right.pta_05123 else "R",
    )


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return centre - half_width, centre + half_width


def cohen_kappa(calls_a: Sequence[str], calls_b: Sequence[str]) -> float | None:
    if len(calls_a) != len(calls_b):
        raise ValueError("Paired call arrays must have the same length")
    if not calls_a:
        return None
    labels = ("L", "R")
    observed = sum(a == b for a, b in zip(calls_a, calls_b, strict=True)) / len(calls_a)
    expected = sum(
        (calls_a.count(label) / len(calls_a)) * (calls_b.count(label) / len(calls_b))
        for label in labels
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else None
    return (observed - expected) / (1.0 - expected)


def summarise_rule(
    pairs: Mapping[str, tuple[EarAudiometry, EarAudiometry]],
    rule: AbnormalityRule,
    mode: str,
) -> tuple[dict[str, object], dict[str, str]]:
    counts = {
        "unilateral_abnormal": 0,
        "bilateral_abnormal": 0,
        "neither_abnormal": 0,
        "index_left": 0,
        "index_right": 0,
    }
    calls: dict[str, str] = {}
    legacy_common = 0
    legacy_agree = 0
    for patient_key, (left, right) in pairs.items():
        state, side = resolve_index_ear(left, right, rule, mode)
        counts[state] += 1
        if side is not None:
            calls[patient_key] = side
            counts["index_left" if side == "L" else "index_right"] += 1
        legacy_side = None
        if left.stage_present != right.stage_present:
            legacy_side = "L" if left.stage_present else "R"
        if side is not None and legacy_side is not None:
            legacy_common += 1
            legacy_agree += int(side == legacy_side)

    total = len(pairs)
    resolved = len(calls)
    unresolved = total - resolved
    resolved_low, resolved_high = wilson_interval(resolved, total)
    legacy_low, legacy_high = wilson_interval(legacy_agree, legacy_common)
    summary: dict[str, object] = {
        "rule_id": rule.rule_id,
        "rule_family": rule.family,
        "rule_label": rule.label,
        "threshold_db": rule.threshold_db,
        "minimum_frequency_count": rule.minimum_frequency_count,
        "bilateral_resolution": mode,
        "complete_paired_patients": total,
        **counts,
        "resolved_index_ear": resolved,
        "unresolved_index_ear": unresolved,
        "resolved_fraction": resolved / total if total else None,
        "resolved_fraction_wilson95_low": resolved_low,
        "resolved_fraction_wilson95_high": resolved_high,
        "legacy_stage_proxy_common": legacy_common,
        "legacy_stage_proxy_agree": legacy_agree,
        "legacy_stage_proxy_agreement": legacy_agree / legacy_common if legacy_common else None,
        "legacy_stage_proxy_agreement_wilson95_low": legacy_low,
        "legacy_stage_proxy_agreement_wilson95_high": legacy_high,
    }
    return summary, calls


def pairwise_agreement(
    calls_by_rule: Mapping[str, Mapping[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rule_a, rule_b in combinations(sorted(calls_by_rule), 2):
        calls_a = calls_by_rule[rule_a]
        calls_b = calls_by_rule[rule_b]
        common = sorted(set(calls_a) & set(calls_b))
        a = [calls_a[key] for key in common]
        b = [calls_b[key] for key in common]
        agreements = sum(x == y for x, y in zip(a, b, strict=True))
        low, high = wilson_interval(agreements, len(common))
        rows.append(
            {
                "rule_a": rule_a,
                "rule_b": rule_b,
                "common_resolved_patients": len(common),
                "same_index_ear": agreements,
                "agreement_fraction": agreements / len(common) if common else None,
                "agreement_fraction_wilson95_low": low,
                "agreement_fraction_wilson95_high": high,
                "cohen_kappa": cohen_kappa(a, b),
            }
        )
    return rows


def consensus_summary(
    patient_keys: Iterable[str],
    calls_by_rule: Mapping[str, Mapping[str, str]],
) -> dict[str, int]:
    result = {
        "all_rules_resolved_and_agree": 0,
        "all_rules_resolved_but_disagree": 0,
        "partial_rules_resolved_and_agree": 0,
        "partial_rules_resolved_and_disagree": 0,
        "no_rule_resolved": 0,
    }
    rule_count = len(calls_by_rule)
    for patient_key in patient_keys:
        calls = [mapping[patient_key] for mapping in calls_by_rule.values() if patient_key in mapping]
        if not calls:
            result["no_rule_resolved"] += 1
        elif len(calls) == rule_count and len(set(calls)) == 1:
            result["all_rules_resolved_and_agree"] += 1
        elif len(calls) == rule_count:
            result["all_rules_resolved_but_disagree"] += 1
        elif len(set(calls)) == 1:
            result["partial_rules_resolved_and_agree"] += 1
        else:
            result["partial_rules_resolved_and_disagree"] += 1
    return result
