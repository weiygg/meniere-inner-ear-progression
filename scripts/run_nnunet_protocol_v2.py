from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run locked-split nnU-Net experiments on LS_SEG_200 only.")
    parser.add_argument("experiment", choices=("E1", "M1", "M2", "M3", "E2", "E4", "E5", "E6"))
    parser.add_argument("--nnunet-raw", type=Path, required=True)
    parser.add_argument("--nnunet-preprocessed", type=Path, required=True)
    parser.add_argument("--nnunet-results", type=Path, required=True)
    parser.add_argument("--dataset", default="Dataset501_LSSemicircularCanals")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Checkpoint interval in epochs. Formal local training uses 1 for robust resume.",
    )
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
        nnUNetTrainerProtocolV2E5,
        nnUNetTrainerProtocolV2E6,
        nnUNetTrainerProtocolV2M2,
        nnUNetTrainerProtocolV2M3,
    )

    trainer_classes = {
        "E1": nnUNetTrainer,
        "M1": nnUNetTrainer,
        "M2": nnUNetTrainerProtocolV2M2,
        "M3": nnUNetTrainerProtocolV2M3,
        "E2": nnUNetTrainer,
        "E4": nnUNetTrainerProtocolV2M2,
        "E5": nnUNetTrainerProtocolV2E5,
        "E6": nnUNetTrainerProtocolV2E6,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(nnUNet_preprocessed) / args.dataset
    split_file = dataset_dir / "splits_final.json"
    plans_file = dataset_dir / f"{args.plans}.json"
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
        "plans": args.plans,
        "batch_size_override": args.batch_size,
        "checkpoint_interval_epochs": args.save_every,
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
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        trainer.batch_size = args.batch_size
    if args.save_every is not None:
        if args.save_every < 1:
            raise ValueError("--save-every must be positive")
        trainer.save_every = args.save_every

    def write_manifest(status: str, **extra: object) -> None:
        manifest["status"] = status
        manifest["last_update_utc"] = datetime.now(timezone.utc).isoformat()
        manifest.update(extra)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def save_interrupt_checkpoint(signum: int, _frame: object) -> None:
        checkpoint = Path(trainer.output_folder) / "checkpoint_latest.pth"
        saved = False
        error = None
        try:
            if getattr(trainer, "was_initialized", False):
                trainer.save_checkpoint(str(checkpoint))
                saved = checkpoint.exists()
        except Exception as exc:  # best effort during a Windows console signal
            error = f"{type(exc).__name__}: {exc}"
        write_manifest(
            "interrupted",
            interrupt_signal=signum,
            checkpoint=str(checkpoint) if saved else None,
            checkpoint_error=error,
        )
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, save_interrupt_checkpoint)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, save_interrupt_checkpoint)
    try:
        maybe_load_checkpoint(trainer, args.continue_training, False, None)
        write_manifest("running", resolved_num_epochs=trainer.num_epochs, save_every=trainer.save_every)
        trainer.run_training()
        trainer.perform_actual_validation(save_probabilities=True)
    except KeyboardInterrupt:
        if manifest.get("status") != "interrupted":
            save_interrupt_checkpoint(int(signal.SIGINT), None)
        return 130
    except Exception as exc:
        write_manifest(
            "failed",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        raise
    write_manifest("complete", completed_utc=datetime.now(timezone.utc).isoformat())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
