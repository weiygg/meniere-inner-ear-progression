from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import nnunetv2.inference.predict_from_raw_data as prediction_module
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from meniere_progression.segmentation.nnunet_trainers import (
    nnUNetTrainerProtocolV2M2,
    nnUNetTrainerProtocolV2M3,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run nnU-Net inference with locally defined Protocol V2 trainer classes."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--fold", default="0")
    parser.add_argument("--checkpoint", default="checkpoint_final.pth")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--disable-tta", action="store_true")
    parser.add_argument("--continue-prediction", action="store_true")
    parser.add_argument("--preprocessing-workers", type=int, default=1)
    parser.add_argument("--export-workers", type=int, default=1)
    args = parser.parse_args()

    classes = {
        nnUNetTrainerProtocolV2M2.__name__: nnUNetTrainerProtocolV2M2,
        nnUNetTrainerProtocolV2M3.__name__: nnUNetTrainerProtocolV2M3,
    }
    original_finder = prediction_module.recursive_find_python_class

    def local_finder(folder: str, class_name: str, current_module: str):
        if class_name in classes:
            return classes[class_name]
        return original_finder(folder, class_name, current_module)

    prediction_module.recursive_find_python_class = local_finder
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=not args.disable_tta,
        perform_everything_on_device=device.type == "cuda",
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        str(args.model_folder), [args.fold], checkpoint_name=args.checkpoint
    )
    predictor.predict_from_files(
        str(args.input_dir),
        str(args.output_dir),
        save_probabilities=False,
        overwrite=not args.continue_prediction,
        num_processes_preprocessing=args.preprocessing_workers,
        num_processes_segmentation_export=args.export_workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
