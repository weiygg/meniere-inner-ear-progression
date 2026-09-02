from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CASE = re.compile(r"^(LSSEG\d+)([LR])$")
EPOCH = re.compile(r"Epoch\s+(\d+)")
LEARNING_RATE = re.compile(r"Current learning rate:\s*([-+0-9.eE]+)")
TRAIN_LOSS = re.compile(r"train_loss\s+([-+0-9.eE]+)")
VAL_LOSS = re.compile(r"val_loss\s+([-+0-9.eE]+)")
PSEUDO_DICE = re.compile(r"Pseudo dice\s+\[(.*)\]")
FLOAT = re.compile(r"(?:np\.float32\()?([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\)?")
EPOCH_TIME = re.compile(r"Epoch time:\s*([-+0-9.eE]+)\s*s")
DATASET = "Dataset502_LSSemicircularCanalsV2"
SPLIT_SHA256 = "0200faaa677123fde897e62e73b9d8753ca10f75d29ee4234f1f92dfa672fd7d"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_training_log(path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "epoch": 0,
        "learning_rate": None,
        "train_loss": None,
        "validation_loss": None,
        "SSC_Dice": None,
        "HSC_Dice": None,
        "PSC_Dice": None,
        "macro_dice": None,
        "best_macro_dice": None,
        "epoch_time_seconds": None,
    }
    if not path.exists():
        return result
    best: float | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := EPOCH.search(line):
            result["epoch"] = int(match.group(1))
        elif match := LEARNING_RATE.search(line):
            result["learning_rate"] = float(match.group(1))
        elif match := TRAIN_LOSS.search(line):
            result["train_loss"] = float(match.group(1))
        elif match := VAL_LOSS.search(line):
            result["validation_loss"] = float(match.group(1))
        elif match := PSEUDO_DICE.search(line):
            values = [float(value) for value in FLOAT.findall(match.group(1))]
            if len(values) == 3:
                result["SSC_Dice"], result["HSC_Dice"], result["PSC_Dice"] = values
                macro = sum(values) / 3
                result["macro_dice"] = macro
                best = macro if best is None else max(best, macro)
        elif match := EPOCH_TIME.search(line):
            result["epoch_time_seconds"] = float(match.group(1))
    result["best_macro_dice"] = best
    return result


def gpu_status() -> dict[str, int | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        values = subprocess.check_output(command, text=True, timeout=10).strip().split(",")
        parsed = [int(value.strip()) for value in values]
        return {
            "utilization_percent": parsed[0],
            "memory_used_mib": parsed[1],
            "memory_total_mib": parsed[2],
            "temperature_c": parsed[3],
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        return {name: None for name in ("utilization_percent", "memory_used_mib", "memory_total_mib", "temperature_c")}


def validate_dataset(raw_root: Path, preprocessed_root: Path, split_csv: Path) -> dict[str, object]:
    import nibabel as nib
    import numpy as np

    raw = raw_root / DATASET
    preprocessed = preprocessed_root / DATASET
    images = sorted((raw / "imagesTr").glob("*.nii.gz"))
    labels = sorted((raw / "labelsTr").glob("*.nii.gz"))
    cases = sorted((preprocessed / "nnUNetPlans_3d_fullres").glob("*.pkl"))
    splits_path = preprocessed / "splits_final.json"
    dataset_json = json.loads((preprocessed / "dataset.json").read_text(encoding="utf-8"))
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    if (len(images), len(labels), len(cases), len(splits)) != (400, 400, 400, 5):
        raise RuntimeError("Dataset502 must contain 400 images, labels and preprocessed ears plus five folds")
    if dataset_json.get("labels") != {"background": 0, "SSC": 1, "HSC": 2, "PSC": 3}:
        raise RuntimeError("Dataset502 label map is not the locked four-class definition")
    label_case_counts = {label: 0 for label in (1, 2, 3)}
    label_voxel_counts = {label: 0 for label in (1, 2, 3)}
    for label_path in labels:
        values = np.asarray(nib.load(str(label_path)).dataobj)
        unique = set(int(value) for value in np.unique(values))
        if not unique <= {0, 1, 2, 3}:
            raise RuntimeError(f"Unexpected label values in {label_path.name}: {sorted(unique)}")
        for label in label_case_counts:
            voxels = int(np.count_nonzero(values == label))
            label_voxel_counts[label] += voxels
            label_case_counts[label] += int(voxels > 0)
    if any(count != 400 for count in label_case_counts.values()):
        raise RuntimeError(f"SSC/HSC/PSC must each be present in all 400 ears: {label_case_counts}")
    validation_cases: list[str] = []
    patient_folds: dict[str, int] = {}
    for fold, split in enumerate(splits):
        train, validation = split["train"], split["val"]
        if len(train) != 320 or len(validation) != 80 or set(train) & set(validation):
            raise RuntimeError(f"Fold {fold} is not the locked 320/80-ear split")
        validation_cases.extend(validation)
        for case in validation:
            match = CASE.fullmatch(case)
            if match is None:
                raise RuntimeError(f"Unexpected case identifier: {case}")
            patient = match.group(1)
            if patient in patient_folds and patient_folds[patient] != fold:
                raise RuntimeError(f"Both ears of {patient} are not in the same fold")
            patient_folds[patient] = fold
    if len(set(validation_cases)) != 400 or len(patient_folds) != 200:
        raise RuntimeError("Every one of 400 ears/200 patients must be OOF exactly once")
    if sha256(splits_path) != SPLIT_SHA256:
        raise RuntimeError("Locked splits_final.json hash mismatch")
    if not split_csv.exists():
        raise FileNotFoundError("Local protected cv_split.csv is missing")
    result = {
        "status": "pass",
        "dataset": DATASET,
        "people": 200,
        "ears": 400,
        "imagesTr": len(images),
        "labelsTr": len(labels),
        "preprocessed_cases": len(cases),
        "folds": len(splits),
        "train_ears_per_fold": 320,
        "validation_ears_per_fold": 80,
        "both_ears_same_fold": True,
        "labels": dataset_json["labels"],
        "label_case_counts": {"SSC": label_case_counts[1], "HSC": label_case_counts[2], "PSC": label_case_counts[3]},
        "label_voxel_counts": {"SSC": label_voxel_counts[1], "HSC": label_voxel_counts[2], "PSC": label_voxel_counts[3]},
        "splits_sha256": SPLIT_SHA256,
        "external_data_loaded": False,
        "checked_utc": now(),
    }
    return result


def model_directory(results_root: Path, plans: str, fold: int) -> Path:
    return results_root / DATASET / f"nnUNetTrainer__{plans}__3d_fullres" / f"fold_{fold}"


def checkpoint_in(folder: Path) -> Path | None:
    for name in ("checkpoint_latest.pth", "checkpoint_final.pth", "checkpoint_best.pth"):
        path = folder / name
        if path.exists():
            return path
    return None


def initial_state(experiment: str, worker_pid: int | None, epoch_cap: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment": experiment,
        "fold": 0,
        "status": "NOT_STARTED",
        "epoch": 0,
        "total_epochs": epoch_cap,
        "native_schedule_epochs": 1000,
        "training_policy": "native_schedule_fixed_compute_cap",
        "metric_scope": "online_validation_pseudo_dice_until_full_fold_validation",
        "current_macro_dice": None,
        "best_macro_dice": None,
        "start_time": now(),
        "last_update": now(),
        "checkpoint": None,
        "pid": None,
        "supervisor_pid": os.getpid(),
        "worker_pid": worker_pid,
        "folds": [
            {
                "fold": fold,
                "status": "NOT_STARTED",
                "epoch": 0,
                "best_macro_dice": None,
                "checkpoint": None,
                "start_time": None,
                "completed_time": None,
            }
            for fold in range(5)
        ],
        "external_data_loaded": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume-safe serial formal five-fold nnU-Net training.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--experiment", default="E1", choices=("E1", "E2"))
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--epoch-cap", type=int, default=54)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--worker-pid", type=int)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        raise ValueError("--poll-seconds must be at least 5")
    if args.epoch_cap < 1 or args.epoch_cap > 1000:
        raise ValueError("--epoch-cap must be between 1 and 1000")

    raw_root = args.run_root / "raw"
    preprocessed_root = args.run_root / "preprocessed"
    results_root = args.run_root / "formal_trained"
    local_results = ROOT / "segmentation_v2" / "results"
    logs = ROOT / "segmentation_v2" / "logs"
    state_path = local_results / "local_training_state.json"
    stop_path = local_results / "stop_requested.json"
    logs.mkdir(parents=True, exist_ok=True)
    local_results.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    stop_path.unlink(missing_ok=True)

    audit = validate_dataset(
        raw_root,
        preprocessed_root,
        ROOT / "segmentation_v2" / "splits" / "cv_split.csv",
    )
    atomic_json(local_results / "local_data_audit.json", audit)
    state = initial_state(args.experiment, args.worker_pid, args.epoch_cap)
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("experiment") == args.experiment:
            for fold in range(5):
                if previous.get("folds", [])[fold].get("status") == "COMPLETED":
                    state["folds"][fold] = previous["folds"][fold]
    state["status"] = "RUNNING"
    atomic_json(state_path, state)

    for fold in range(5):
        fold_state = state["folds"][fold]
        folder = model_directory(results_root, args.plans, fold)
        validation_summary = folder / "validation" / "summary.json"
        if fold_state.get("status") == "COMPLETED" and validation_summary.exists():
            continue
        checkpoint = checkpoint_in(folder)
        fold_log = logs / f"nnunet_fold{fold}.log"
        monitor_log = logs / f"nnunet_fold{fold}_monitor.ndjson"
        run_dir = local_results / "formal_e1" / f"fold_{fold}"
        command = [
            str(args.python),
            "-u",
            str(ROOT / "scripts" / "run_nnunet_protocol_v2.py"),
            args.experiment,
            "--nnunet-raw",
            str(raw_root),
            "--nnunet-preprocessed",
            str(preprocessed_root),
            "--nnunet-results",
            str(results_root),
            "--dataset",
            DATASET,
            "--fold",
            str(fold),
            "--plans",
            args.plans,
            "--device",
            "cuda",
            "--run-dir",
            str(run_dir),
            "--save-every",
            "1",
            "--native-schedule-epoch-cap",
            str(args.epoch_cap),
        ]
        if checkpoint is not None:
            command.append("--continue-training")
        fold_state.update(
            {
                "status": "RUNNING",
                "start_time": fold_state.get("start_time") or now(),
                "checkpoint": str(checkpoint) if checkpoint else None,
                "log": str(fold_log),
            }
        )
        state.update({"fold": fold, "status": "RUNNING", "last_update": now()})
        atomic_json(state_path, state)
        environment = os.environ.copy()
        environment.update(
            {
                "nnUNet_raw": str(raw_root),
                "nnUNet_preprocessed": str(preprocessed_root),
                "nnUNet_results": str(results_root),
            }
        )
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        with fold_log.open("a", encoding="utf-8", buffering=1) as output:
            output.write(f"\n{now()} FORMAL SERIAL TRAINING COMMAND: {json.dumps(command)}\n")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags,
            )
            state["pid"] = process.pid
            fold_state["pid"] = process.pid
            atomic_json(state_path, state)
            print(f"Formal {args.experiment} fold {fold} started: training PID {process.pid}", flush=True)
            while process.poll() is None:
                parsed = parse_training_log(fold_log)
                gpu = gpu_status()
                checkpoint = checkpoint_in(folder)
                fold_state.update(
                    {
                        "epoch": parsed["epoch"],
                        "best_macro_dice": parsed["best_macro_dice"],
                        "checkpoint": str(checkpoint) if checkpoint else None,
                        "last_update": now(),
                    }
                )
                state.update(
                    {
                        "epoch": parsed["epoch"],
                        "current_macro_dice": parsed["macro_dice"],
                        "best_macro_dice": parsed["best_macro_dice"],
                        "checkpoint": str(checkpoint) if checkpoint else None,
                        "last_update": now(),
                        "gpu": gpu,
                        "latest_metrics": parsed,
                    }
                )
                atomic_json(state_path, state)
                with monitor_log.open("a", encoding="utf-8") as monitor:
                    monitor.write(json.dumps({"time": now(), "pid": process.pid, "metrics": parsed, "gpu": gpu}) + "\n")
                print(
                    f"fold={fold} epoch={parsed['epoch']} macro={parsed['macro_dice']} "
                    f"gpu={gpu['utilization_percent']}% memory={gpu['memory_used_mib']}MiB",
                    flush=True,
                )
                if stop_path.exists():
                    print("Stop requested; sending CTRL_BREAK to the exact training process.", flush=True)
                    try:
                        process.send_signal(signal.CTRL_BREAK_EVENT)
                    except (AttributeError, OSError):
                        process.terminate()
                time.sleep(args.poll_seconds)
        return_code = process.returncode
        state["pid"] = None
        fold_state["pid"] = None
        if stop_path.exists() or return_code == 130:
            checkpoint = checkpoint_in(folder)
            fold_state.update({"status": "INTERRUPTED", "checkpoint": str(checkpoint) if checkpoint else None})
            state.update({"status": "INTERRUPTED", "checkpoint": str(checkpoint) if checkpoint else None, "last_update": now()})
            atomic_json(state_path, state)
            return 130
        if return_code != 0:
            fold_state["status"] = "FAILED"
            state.update({"status": "FAILED", "last_update": now(), "return_code": return_code})
            atomic_json(state_path, state)
            return return_code

        prediction_dir = folder / "validation"
        case_csv = local_results / f"nnunet_fold{fold}_case_metrics.csv"
        summary_csv = local_results / f"nnunet_fold{fold}_metrics.csv"
        summary_json = local_results / f"nnunet_fold{fold}_metrics.json"
        evaluation = [
            str(args.python),
            str(ROOT / "segmentation_v2" / "scripts" / "evaluate_nnunet_fold.py"),
            "--prediction-dir",
            str(prediction_dir),
            "--ground-truth-dir",
            str(preprocessed_root / DATASET / "gt_segmentations"),
            "--fold",
            str(fold),
            "--case-output",
            str(case_csv),
            "--summary-output",
            str(summary_csv),
            "--json-output",
            str(summary_json),
            "--oof-dir",
            str(local_results / "oof_predictions" / f"fold_{fold}"),
        ]
        subprocess.run(evaluation, cwd=ROOT, check=True)
        fold_summary = json.loads(summary_json.read_text(encoding="utf-8"))
        fold_state.update(
            {
                "status": "COMPLETED",
                "epoch": args.epoch_cap,
                "best_macro_dice": fold_summary["macro_dice"],
                "checkpoint": str(folder / "checkpoint_final.pth"),
                "completed_time": now(),
                "metrics": str(summary_csv),
            }
        )
        state.update({"current_macro_dice": fold_summary["macro_dice"], "last_update": now()})
        atomic_json(state_path, state)

    oof_command = [
        str(args.python),
        str(ROOT / "segmentation_v2" / "scripts" / "summarize_oof_metrics.py"),
        "--case-metrics",
        *[str(local_results / f"nnunet_fold{fold}_case_metrics.csv") for fold in range(5)],
        "--output-csv",
        str(local_results / "nnunet_oof_metrics.csv"),
        "--output-md",
        str(local_results / "nnunet_oof_summary.md"),
        "--output-json",
        str(local_results / "nnunet_oof_summary.json"),
        "--bootstrap",
        "5000",
        "--seed",
        "20260830",
    ]
    subprocess.run(oof_command, cwd=ROOT, check=True)
    oof = json.loads((local_results / "nnunet_oof_summary.json").read_text(encoding="utf-8"))
    state.update(
        {
            "status": "COMPLETED",
            "fold": 4,
            "epoch": args.epoch_cap,
            "current_macro_dice": oof["metrics"]["Macro"]["dice"]["estimate"],
            "best_macro_dice": oof["metrics"]["Macro"]["dice"]["estimate"],
            "pid": None,
            "last_update": now(),
            "completed_time": now(),
            "stage_b_residual_recommended": oof["metrics"]["Macro"]["dice"]["estimate"] < 0.83,
        }
    )
    atomic_json(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
