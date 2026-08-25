from __future__ import annotations

from meniere_progression.hearing_ear_rules import (
    EarAudiometry,
    build_rule_grid,
    cohen_kappa,
    resolve_index_ear,
    summarise_rule,
    wilson_interval,
)


def ear(side: str, values: tuple[float, float, float, float]) -> EarAudiometry:
    return EarAudiometry(side, values)


def test_rule_grid_has_all_requested_combinations() -> None:
    rules = build_rule_grid()
    assert len(rules) == 30
    assert len({rule.rule_id for rule in rules}) == 30


def test_strict_rule_resolves_only_unilateral_abnormality() -> None:
    rule = next(rule for rule in build_rule_grid() if rule.rule_id == "pta_05123_gt_25")
    state, side = resolve_index_ear(
        ear("L", (40, 40, 40, 40)),
        ear("R", (10, 10, 10, 10)),
        rule,
        "strict",
    )
    assert (state, side) == ("unilateral_abnormal", "L")

    state, side = resolve_index_ear(
        ear("L", (40, 40, 40, 40)),
        ear("R", (35, 35, 35, 35)),
        rule,
        "strict",
    )
    assert (state, side) == ("bilateral_abnormal", None)


def test_bilateral_gap_rule_requires_prespecified_difference() -> None:
    rule = next(rule for rule in build_rule_grid() if rule.rule_id == "pta_05123_gt_25")
    left = ear("L", (50, 50, 50, 50))
    right = ear("R", (38, 38, 38, 38))
    assert resolve_index_ear(left, right, rule, "pta_gap_10") == ("bilateral_abnormal", "L")
    assert resolve_index_ear(left, right, rule, "pta_gap_15") == ("bilateral_abnormal", None)


def test_equal_bilateral_pta_remains_unresolved_even_in_worse_ear_comparator() -> None:
    rule = next(rule for rule in build_rule_grid() if rule.rule_id == "pta_05123_gt_25")
    left = ear("L", (40, 40, 40, 40))
    right = ear("R", (40, 40, 40, 40))
    assert resolve_index_ear(left, right, rule, "worse_pta_no_minimum") == (
        "bilateral_abnormal",
        None,
    )


def test_interval_and_kappa_are_bounded() -> None:
    low, high = wilson_interval(8, 10)
    assert low is not None and high is not None and 0 <= low < high <= 1
    assert cohen_kappa(["L", "R", "L", "R"], ["L", "R", "L", "R"]) == 1.0


def test_summary_counts_left_and_right_calls() -> None:
    rule = next(rule for rule in build_rule_grid() if rule.rule_id == "pta_05123_gt_25")
    pairs = {
        "a": (ear("L", (40, 40, 40, 40)), ear("R", (10, 10, 10, 10))),
        "b": (ear("L", (10, 10, 10, 10)), ear("R", (40, 40, 40, 40))),
    }
    summary, calls = summarise_rule(pairs, rule, "strict")
    assert summary["index_left"] == 1
    assert summary["index_right"] == 1
    assert calls == {"a": "L", "b": "R"}
