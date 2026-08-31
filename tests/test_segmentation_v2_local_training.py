from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_local_runner():
    path = ROOT / "segmentation_v2" / "scripts" / "run_local_5fold.py"
    spec = importlib.util.spec_from_file_location("segmentation_v2_local_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_training_log_parser_reports_online_metrics(tmp_path: Path) -> None:
    module = load_local_runner()
    log = tmp_path / "fold.log"
    log.write_text(
        "2026-08-31: Epoch 7\n"
        "2026-08-31: Current learning rate: 0.00994\n"
        "2026-08-31: train_loss -0.42\n"
        "2026-08-31: val_loss -0.51\n"
        "2026-08-31: Pseudo dice [np.float32(0.70), np.float32(0.80), np.float32(0.75)]\n"
        "2026-08-31: Epoch time: 123.4 s\n",
        encoding="utf-8",
    )
    parsed = module.parse_training_log(log)
    assert parsed["epoch"] == 7
    assert parsed["macro_dice"] == 0.75
    assert parsed["learning_rate"] == 0.00994
    assert parsed["epoch_time_seconds"] == 123.4


def test_formal_local_config_keeps_nnunet_default_schedule() -> None:
    config = yaml.safe_load(
        (ROOT / "segmentation_v2" / "configs" / "local_e1_formal.yaml").read_text(encoding="utf-8")
    )
    assert config["schedule"]["epochs"] == 1000
    assert config["schedule"]["screening"] is False
    assert config["checkpoint"]["interval_epochs"] == 1
    assert config["selection"]["external_labels_allowed"] is False


def test_detached_scripts_are_project_pid_scoped() -> None:
    start = (ROOT / "segmentation_v2" / "run_local_training.ps1").read_text(encoding="utf-8")
    stop = (ROOT / "segmentation_v2" / "stop_training.ps1").read_text(encoding="utf-8")
    resume = (ROOT / "segmentation_v2" / "resume_training.ps1").read_text(encoding="utf-8")
    assert "Start-Process" in start and "-WindowStyle Normal" in start
    assert "Stop-Process -Id $exactPid" in stop
    assert "Stop-Process -Name" not in stop
    assert "run_local_training.ps1" in resume
