from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_center_split_is_disjoint_and_external_sets_are_frozen() -> None:
    config = yaml.safe_load((ROOT / "configs/center_split.yaml").read_text(encoding="utf-8"))
    primary = set(config["primary_development"]["source_groups"])
    external_1 = set(config["external_validation_1"]["source_groups"])
    external_2 = set(config["external_validation_2"]["source_groups"])
    assert not (primary & external_1 or primary & external_2 or external_1 & external_2)
    assert config["split_unit"] == "patient"
    assert config["external_validation_1"]["role"] == "frozen_external_test_only"
    assert config["external_validation_2"]["role"] == "frozen_external_test_only"
    assert config["constraints"]["external_data_used_for_training"] is False
    assert config["constraints"]["external_data_used_for_model_selection"] is False
    assert config["constraints"]["external_data_used_for_threshold_selection"] is False
