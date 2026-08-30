from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import nnunetv2.inference.predict_from_raw_data as prediction_module
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from meniere_progression.segmentation.nnunet_trainers import (
    nnUNetTrainerProtocolV2M2,
    nnUNetTrainerProtocolV2M3,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Five-fold probability-average nnU-Net inference.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=(0, 1, 2, 3, 4))
    parser.add_argument("--checkpoint", default="checkpoint_final.pth")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--disable-tta", action="store_true")
    parser.add_argument("--save-probabilities", action="store_true")
    args = parser.parse_args()
    if sorted(args.folds) != [0, 1, 2, 3, 4]:
        raise ValueError("The final ensemble requires exactly folds 0-4")
    classes = {cls.__name__: cls for cls in (nnUNetTrainerProtocolV2M2, nnUNetTrainerProtocolV2M3)}
    original_finder = prediction_module.recursive_find_python_class

    def local_finder(folder: str, class_name: str, current_module: str):
        return classes.get(class_name) or original_finder(folder, class_name, current_module)

    prediction_module.recursive_find_python_class = local_finder
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
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
    predictor.initialize_from_trained_model_folder(str(args.model_folder), args.folds, checkpoint_name=args.checkpoint)
    predictor.predict_from_files(
        str(args.input_dir),
        str(args.output_dir),
        save_probabilities=args.save_probabilities,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
