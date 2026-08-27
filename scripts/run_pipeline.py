from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meniere_progression.registry import DatasetRegistry, validate_overlap_file
from meniere_progression.reporting.manifests import build_run_manifest, write_run_manifest


PHASES = (
    "registry",
    "clinical_qc",
    "segmentation_redevelopment",
    "segmentation_evaluation",
    "geometry_reliability",
    "clinical_feature_build",
    "pebm_eligibility",
    "pebm_primary",
    "secondary_prediction",
    "report",
)


def phase_blockers(phase: str, config_dir: Path) -> list[str]:
    clinical = yaml.safe_load((config_dir / "clinical_codebook.yaml").read_text(encoding="utf-8"))
    pebm = yaml.safe_load((config_dir / "pebm_analysis.yaml").read_text(encoding="utf-8"))
    blockers: list[str] = []
    if phase in {"clinical_feature_build", "pebm_eligibility", "pebm_primary", "secondary_prediction"}:
        if clinical["status"] != "signed":
            blockers.append("clinical_codebook_not_signed")
    if phase in {"pebm_primary", "secondary_prediction"}:
        blockers.extend(name for name, value in pebm["blocking_gates"].items() if not value)
    if phase == "secondary_prediction":
        blockers.append("secondary_prediction_requires_frozen_P_EBM_stage_and_signed_outcome")
    return sorted(set(blockers))


def main() -> int:
    parser = argparse.ArgumentParser(description="Protocol V2 phase-gated pipeline runner.")
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/runs")
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    registry = DatasetRegistry.load(args.config_dir / "dataset_registry.yaml")
    registry.validate()
    validate_overlap_file(args.config_dir / "dataset_overlap.yaml")
    blockers = phase_blockers(args.phase, args.config_dir)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{args.phase}"
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    status_path = run_dir / "phase_status.json"
    status_path.write_text(
        json.dumps({"phase": args.phase, "status": "blocked" if blockers else "ready", "blockers": blockers}, indent=2),
        encoding="utf-8",
    )
    config_paths = sorted(args.config_dir.glob("*.yaml"))
    manifest = build_run_manifest(
        root=ROOT,
        run_id=run_id,
        phase=args.phase,
        config_paths=config_paths,
        data_manifest_paths=[],
        random_seeds={"global": args.seed},
        outputs=[status_path],
        blockers=blockers,
    )
    write_run_manifest(run_dir / "run_manifest.json", manifest)
    print(json.dumps({"run_id": run_id, "status": "blocked" if blockers else "ready", "blockers": blockers}, indent=2))
    return 2 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
