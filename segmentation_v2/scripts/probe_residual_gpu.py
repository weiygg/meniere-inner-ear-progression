from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe one mixed-precision forward/backward pass for Residual Encoder nnU-Net."
    )
    parser.add_argument("--nnunet-raw", type=Path, required=True)
    parser.add_argument("--nnunet-preprocessed", type=Path, required=True)
    parser.add_argument("--nnunet-results", type=Path, required=True)
    parser.add_argument("--dataset", default="Dataset502_LSSemicircularCanalsV2")
    parser.add_argument("--plans", default="nnUNetResEncUNetMPlans.json")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GPU probe")

    os.environ["nnUNet_raw"] = str(args.nnunet_raw.resolve())
    os.environ["nnUNet_preprocessed"] = str(args.nnunet_preprocessed.resolve())
    os.environ["nnUNet_results"] = str(args.nnunet_results.resolve())
    # nnU-Net reads its roots when imported, so import only after setting them.
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    dataset_dir = args.nnunet_preprocessed / args.dataset
    plans = load_json(dataset_dir / args.plans)
    dataset_json = load_json(dataset_dir / "dataset.json")
    configuration = plans["configurations"]["3d_fullres"]
    patch_size = tuple(int(value) for value in configuration["patch_size"])
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    status = "pass"
    error = None
    parameters = None
    try:
        trainer = nnUNetTrainer(plans, "3d_fullres", 0, dataset_json, device=device)
        trainer.initialize()
        network = trainer.network
        network.train()
        parameters = sum(parameter.numel() for parameter in network.parameters() if parameter.requires_grad)
        sample = torch.randn((args.batch_size, 1, *patch_size), device=device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = network(sample)
            if not isinstance(outputs, (tuple, list)):
                outputs = [outputs]
            loss = sum(output.float().square().mean() for output in outputs)
        loss.backward()
        output_shapes = [list(output.shape) for output in outputs]
        del loss, outputs, sample, network, trainer
    except torch.cuda.OutOfMemoryError as exc:
        status = "oom"
        error = str(exc)
        output_shapes = []
    peak_bytes = int(torch.cuda.max_memory_allocated(device))
    result = {
        "status": status,
        "dataset": args.dataset,
        "plans": args.plans,
        "architecture": configuration["architecture"]["network_class_name"],
        "planned_batch_size": int(configuration["batch_size"]),
        "probed_batch_size": args.batch_size,
        "patch_size_dhw": list(patch_size),
        "trainable_parameters": int(parameters) if parameters is not None else None,
        "mixed_precision": "float16 autocast; float32 scalar loss",
        "forward_backward_completed": status == "pass",
        "output_shapes": output_shapes,
        "peak_memory_allocated_bytes": peak_bytes,
        "peak_memory_allocated_gib": peak_bytes / 1024**3,
        "gpu": torch.cuda.get_device_name(device),
        "external_data_loaded": False,
        "error": error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    torch.cuda.empty_cache()
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
