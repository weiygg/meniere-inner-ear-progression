from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


REQUIRED_SENTENCE = "Model frozen before external evaluation. No external annotation was used for model selection."


def main() -> int:
    parser = argparse.ArgumentParser(description="Guard a single frozen external evaluation command.")
    parser.add_argument("--final-config", type=Path, required=True)
    parser.add_argument("--freeze-document", type=Path, required=True)
    parser.add_argument("--sentinel", type=Path, required=True)
    parser.add_argument("--confirm-final-evaluation", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config = yaml.safe_load(args.final_config.read_text(encoding="utf-8"))
    freeze_text = args.freeze_document.read_text(encoding="utf-8")
    blockers = []
    if config.get("status") != "frozen":
        blockers.append("final_config_not_frozen")
    if config.get("external_evaluation_allowed") is not True:
        blockers.append("external_evaluation_not_allowed")
    if REQUIRED_SENTENCE not in freeze_text:
        blockers.append("freeze_statement_missing")
    if args.sentinel.exists():
        blockers.append("external_evaluation_already_completed")
    if not args.confirm_final_evaluation:
        blockers.append("explicit_confirmation_missing")
    if not args.command:
        blockers.append("evaluation_command_missing")
    if blockers:
        print(json.dumps({"status": "blocked", "blockers": blockers}))
        return 2
    completed = subprocess.run(args.command, check=False)
    if completed.returncode != 0:
        return completed.returncode
    args.sentinel.parent.mkdir(parents=True, exist_ok=True)
    args.sentinel.write_text(
        json.dumps({"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat()}) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
