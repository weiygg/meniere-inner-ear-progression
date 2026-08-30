from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one or all patient-level Dataset502 nnU-Net folds.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", choices=("0", "1", "2", "3", "4", "all"), required=True)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--experiment", choices=("M1", "M2", "M3", "E2", "E4", "E5", "E6"), default="M1")
    parser.add_argument("--nnunet-raw", type=Path)
    parser.add_argument("--nnunet-preprocessed", type=Path)
    parser.add_argument("--nnunet-results", type=Path)
    parser.add_argument("--run-root", type=Path, default=ROOT / "segmentation_v2" / "results" / "training")
    parser.add_argument("--continue-training", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["model_selection"]["external_labels_allowed"] is not False:
        raise ValueError("External labels must remain forbidden for model selection")
    if args.num_epochs < 1:
        raise ValueError("--num-epochs must be positive")
    roots = (args.nnunet_raw, args.nnunet_preprocessed, args.nnunet_results)
    if not args.dry_run and any(path is None for path in roots):
        raise ValueError("All three nnU-Net roots are required unless --dry-run is used")
    folds = range(5) if args.fold == "all" else [int(args.fold)]
    plans = str(config.get("plans", "nnUNetPlans"))
    runtime_batch_size = config.get("training", {}).get("runtime_batch_size_on_6gb_gpu")
    commands = []
    for fold in folds:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_nnunet_protocol_v2.py"),
            args.experiment,
            "--nnunet-raw",
            str(args.nnunet_raw or "<NNUNET_RAW>"),
            "--nnunet-preprocessed",
            str(args.nnunet_preprocessed or "<NNUNET_PREPROCESSED>"),
            "--nnunet-results",
            str(args.nnunet_results or "<NNUNET_RESULTS>"),
            "--dataset",
            f"Dataset{int(config['dataset_id']):03d}_{config['dataset_name']}",
            "--fold",
            str(fold),
            "--device",
            "cuda",
            "--run-dir",
            str(args.run_root / f"{args.experiment.lower()}_fold{fold}_e{args.num_epochs}"),
            "--num-epochs",
            str(args.num_epochs),
            "--plans",
            plans,
        ]
        if runtime_batch_size is not None:
            command.extend(["--batch-size", str(int(runtime_batch_size))])
        if args.continue_training:
            command.append("--continue-training")
        commands.append(command)
        if not args.dry_run:
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                return completed.returncode
    print(json.dumps({"status": "dry_run" if args.dry_run else "complete", "commands": commands}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
