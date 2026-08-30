from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_external_gate_blocks_unfrozen_v2(tmp_path: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "segmentation_v2" / "scripts" / "external_eval.py"),
        "--final-config",
        str(ROOT / "segmentation_v2" / "configs" / "final.yaml"),
        "--freeze-document",
        str(ROOT / "segmentation_v2" / "MODEL_FREEZE.md"),
        "--sentinel",
        str(tmp_path / "external.completed"),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert completed.returncode == 2
    result = json.loads(completed.stdout)
    assert result["status"] == "blocked"
    assert "final_config_not_frozen" in result["blockers"]
    assert not (tmp_path / "external.completed").exists()


def test_residual_five_fold_dry_run_is_locked() -> None:
    command = [
        sys.executable,
        str(ROOT / "segmentation_v2" / "scripts" / "train_cv.py"),
        "--config",
        str(ROOT / "segmentation_v2" / "configs" / "residual_nnunet.yaml"),
        "--fold",
        "all",
        "--num-epochs",
        "5",
        "--experiment",
        "E2",
        "--dry-run",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    result = json.loads(completed.stdout)
    assert len(result["commands"]) == 5
    for fold, planned in enumerate(result["commands"]):
        assert planned[planned.index("--fold") + 1] == str(fold)
        assert planned[planned.index("--plans") + 1] == "nnUNetResEncUNetMPlans"
        assert planned[planned.index("--batch-size") + 1] == "1"


def test_public_v2_manifests_are_aggregate_only() -> None:
    manifest_dir = ROOT / "data" / "manifests"
    paths = sorted(manifest_dir.glob("segmentation_v2_*.json"))
    assert paths
    forbidden_keys = {"patient", "patient_id", "ear", "image_path", "mask_path", "prediction_path"}

    def visit(value: object) -> None:
        if isinstance(value, dict):
            assert forbidden_keys.isdisjoint(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for path in paths:
        visit(json.loads(path.read_text(encoding="utf-8")))
