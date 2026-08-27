from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M1/M2/M3 on the locked internal split only.")
    parser.add_argument("experiment", choices=("M1", "M2", "M3"))
    parser.add_argument("--nnunet-raw", type=Path, required=True)
    parser.add_argument("--nnunet-preprocessed", type=Path, required=True)
    parser.add_argument("--nnunet-results", type=Path, required=True)
    parser.add_argument("--dataset", default="Dataset501_LSSemicircularCanals")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--continue-training", action="store_true")
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Optional fixed equal-budget pilot cap. Omit for the nnU-Net default.",
    )
    args = parser.parse_args()

    os.environ["nnUNet_raw"] = str(args.nnunet_raw.resolve())
    os.environ["nnUNet_preprocessed"] = str(args.nnunet_preprocessed.resolve())
    os.environ["nnUNet_results"] = str(args.nnunet_results.resolve())

    import torch
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.run.run_training import maybe_load_checkpoint
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from meniere_progression.segmentation.nnunet_trainers import (
        nnUNetTrainerProtocolV2M2,
        nnUNetTrainerProtocolV2M3,
    )

    trainer_classes = {
        "M1": nnUNetTrainer,
        "M2": nnUNetTrainerProtocolV2M2,
        "M3": nnUNetTrainerProtocolV2M3,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(nnUNet_preprocessed) / args.dataset
    split_file = dataset_dir / "splits_final.json"
    plans_file = dataset_dir / "nnUNetPlans.json"
    if not split_file.exists() or not plans_file.exists():
        raise FileNotFoundError("Planning outputs or locked splits are missing")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    manifest = {
        "status": "starting",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": args.experiment,
        "dataset": args.dataset,
        "configuration": "3d_fullres",
        "fold": args.fold,
        "trainer": trainer_classes[args.experiment].__name__,
        "model_selection_source": "LS_SEG_200_validation_only",
        "external_labels_loaded": False,
        "run_mode": "equal_budget_internal_pilot" if args.num_epochs is not None else "full_default",
        "num_epochs": args.num_epochs,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    manifest_path = args.run_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    trainer = trainer_classes[args.experiment](
        plans=load_json(plans_file),
        configuration="3d_fullres",
        fold=args.fold,
        dataset_json=load_json(dataset_dir / "dataset.json"),
        device=device,
    )
    if args.num_epochs is not None:
        if args.num_epochs < 1:
            raise ValueError("--num-epochs must be positive")
        trainer.num_epochs = args.num_epochs
    maybe_load_checkpoint(trainer, args.continue_training, False, None)
    trainer.run_training()
    trainer.perform_actual_validation(save_probabilities=True)
    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
