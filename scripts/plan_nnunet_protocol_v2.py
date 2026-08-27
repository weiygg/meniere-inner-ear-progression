from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan/preprocess locked Protocol V2 nnU-Net M1 data.")
    parser.add_argument("--nnunet-raw", type=Path, required=True)
    parser.add_argument("--nnunet-preprocessed", type=Path, required=True)
    parser.add_argument("--nnunet-results", type=Path, required=True)
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--dataset-name", default="LSSemicircularCanals")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    template = args.nnunet_raw / dataset / "splits_final.json.template"
    if not template.exists():
        raise FileNotFoundError(f"Missing locked split template: {template}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "nnUNet_raw": str(args.nnunet_raw.resolve()),
            "nnUNet_preprocessed": str(args.nnunet_preprocessed.resolve()),
            "nnUNet_results": str(args.nnunet_results.resolve()),
        }
    )
    command = [
        "nnUNetv2_plan_and_preprocess",
        "-d",
        str(args.dataset_id),
        "-c",
        "3d_fullres",
        "-np",
        "1",
        "--verify_dataset_integrity",
    ]
    completed = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    (args.run_dir / "planner_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (args.run_dir / "planner_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"nnU-Net planning failed with exit code {completed.returncode}")

    preprocessed_dataset = args.nnunet_preprocessed / dataset
    split_target = preprocessed_dataset / "splits_final.json"
    shutil.copy2(template, split_target)
    plan_candidates = sorted(preprocessed_dataset.glob("*Plans.json"))
    fingerprint = preprocessed_dataset / "dataset_fingerprint.json"
    if not plan_candidates or not fingerprint.exists():
        raise RuntimeError("Planner exited successfully but plans/fingerprint are missing")
    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "configuration": "3d_fullres",
        "planner_command": command,
        "planner_returncode": completed.returncode,
        "preprocessing_workers": 1,
        "fingerprint_sha256": sha256(fingerprint),
        "plans": {path.name: sha256(path) for path in plan_candidates},
        "locked_split_sha256": sha256(split_target),
        "split_policy": "280_train_60_validation; 60_internal_benchmark_excluded",
    }
    (args.run_dir / "planning_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
