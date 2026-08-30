from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from batchgenerators.utilities.file_and_folder_operations import load_json


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


CASE = re.compile(r"^(LSSEG\d+)([LR])$")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test Dataset502 splits and nnU-Net trainers.")
    parser.add_argument("--nnunet-raw", type=Path, required=True)
    parser.add_argument("--nnunet-preprocessed", type=Path, required=True)
    parser.add_argument("--nnunet-results", type=Path, required=True)
    parser.add_argument("--dataset", default="Dataset502_LSSemicircularCanalsV2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    os.environ["nnUNet_raw"] = str(args.nnunet_raw.resolve())
    os.environ["nnUNet_preprocessed"] = str(args.nnunet_preprocessed.resolve())
    os.environ["nnUNet_results"] = str(args.nnunet_results.resolve())
    # nnU-Net resolves its data roots at import time, so trainer imports must
    # occur only after the three environment variables are set.
    from meniere_progression.segmentation.nnunet_trainers import (
        nnUNetTrainerProtocolV2M2,
        nnUNetTrainerProtocolV2M3,
    )
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    dataset = args.nnunet_preprocessed / args.dataset
    splits = load_json(dataset / "splits_final.json")
    plans = load_json(dataset / "nnUNetPlans.json")
    dataset_json = load_json(dataset / "dataset.json")
    if len(splits) != 5:
        raise ValueError("Expected five folds")
    validation_cases: list[str] = []
    for fold, split in enumerate(splits):
        if len(split["train"]) != 320 or len(split["val"]) != 80:
            raise ValueError(f"Fold {fold} does not contain 320 train and 80 validation ears")
        if set(split["train"]) & set(split["val"]):
            raise ValueError(f"Fold {fold} has train-validation overlap")
        validation_cases.extend(split["val"])
    if len(validation_cases) != 400 or len(set(validation_cases)) != 400:
        raise ValueError("Each ear must appear in exactly one validation fold")
    patient_fold: dict[str, int] = {}
    for fold, split in enumerate(splits):
        for case in split["val"]:
            match = CASE.fullmatch(case)
            if match is None:
                raise ValueError(f"Unexpected case ID: {case}")
            patient = match.group(1)
            if patient in patient_fold and patient_fold[patient] != fold:
                raise ValueError(f"Patient ears cross folds: {patient}")
            patient_fold[patient] = fold
    if len(patient_fold) != 200:
        raise ValueError("Expected 200 validation patients across folds")
    preprocessed_dir = dataset / plans["configurations"]["3d_fullres"]["data_identifier"]
    # nnU-Net 2.6 uses one .pkl metadata file per case and Blosc2 arrays; older
    # releases used .npz. Count the stable per-case metadata files.
    preprocessed_cases = list(preprocessed_dir.glob("*.pkl"))
    if len(preprocessed_cases) != 400:
        raise ValueError(f"Expected 400 preprocessed cases, found {len(preprocessed_cases)}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    trainer_rows = []
    for trainer_class in (nnUNetTrainer, nnUNetTrainerProtocolV2M2, nnUNetTrainerProtocolV2M3):
        trainer = trainer_class(plans, "3d_fullres", 0, dataset_json, device=device)
        trainer.initialize()
        parameters = sum(parameter.numel() for parameter in trainer.network.parameters() if parameter.requires_grad)
        loss_name = type(trainer.loss).__name__
        output_channels = int(trainer.network.decoder.seg_layers[0].out_channels)
        trainer_rows.append(
            {
                "trainer": trainer_class.__name__,
                "trainable_parameters": int(parameters),
                "loss": loss_name,
                "output_channels": output_channels,
            }
        )
        del trainer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {
        "status": "pass",
        "dataset": args.dataset,
        "people": 200,
        "ears": 400,
        "folds": 5,
        "both_ears_same_fold": True,
        "each_ear_is_oof_once": True,
        "preprocessed_cases": 400,
        "patch_size": plans["configurations"]["3d_fullres"]["patch_size"],
        "spacing": plans["configurations"]["3d_fullres"]["spacing"],
        "trainers": trainer_rows,
        "device": str(device),
        "external_data_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
