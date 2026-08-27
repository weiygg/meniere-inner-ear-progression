from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_MANIFEST_FIELDS = {
    "run_id",
    "created_at",
    "phase",
    "git_commit",
    "config_hashes",
    "data_manifest_hashes",
    "package_versions",
    "random_seeds",
    "outputs",
    "warnings",
    "blockers",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def package_versions(names: tuple[str, ...] = ("numpy", "pandas", "scipy", "PyYAML")) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not_installed"
    return result


def build_run_manifest(
    *,
    root: Path,
    run_id: str,
    phase: str,
    config_paths: list[Path],
    data_manifest_paths: list[Path],
    random_seeds: dict[str, int],
    outputs: list[Path],
    warnings: list[str] | None = None,
    blockers: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "git_commit": _git_commit(root),
        "config_hashes": {path.name: _sha256(path) for path in config_paths},
        "data_manifest_hashes": {path.name: _sha256(path) for path in data_manifest_paths},
        "package_versions": package_versions(),
        "random_seeds": random_seeds,
        "outputs": [
            {"name": path.name, "sha256": _sha256(path)} for path in outputs if path.exists()
        ],
        "warnings": warnings or [],
        "blockers": blockers or [],
    }


def validate_run_manifest(payload: dict[str, Any]) -> None:
    missing = REQUIRED_MANIFEST_FIELDS - set(payload)
    if missing:
        raise ValueError(f"Run manifest missing fields: {sorted(missing)}")


def write_run_manifest(path: Path, payload: dict[str, Any]) -> None:
    validate_run_manifest(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
