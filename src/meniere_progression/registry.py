from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ProtocolViolation


REQUIRED_DATASETS = {
    "LS_SEG_200",
    "Z2_SEG_EXT1",
    "Z2_SEG_EXT2",
    "EXT_MANUAL_50",
    "LS_CLIN_79",
    "Z2_CLIN",
}
SELECTION_OPERATIONS = {
    "training": "training_allowed",
    "model_selection": "model_selection_allowed",
    "threshold_selection": "threshold_selection_allowed",
    "feature_selection": "feature_selection_allowed",
}


@dataclass(frozen=True)
class DatasetRegistry:
    payload: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "DatasetRegistry":
        return cls(yaml.safe_load(path.read_text(encoding="utf-8")))

    @property
    def datasets(self) -> dict[str, dict[str, Any]]:
        return self.payload["datasets"]

    def validate(self) -> None:
        missing = REQUIRED_DATASETS - set(self.datasets)
        if missing:
            raise ProtocolViolation(f"Dataset registry missing: {sorted(missing)}")
        if self.datasets["LS_SEG_200"].get("people") != 200:
            raise ProtocolViolation("LS_SEG_200 must contain 200 people")
        if self.datasets["LS_CLIN_79"].get("people") != 79:
            raise ProtocolViolation("LS_CLIN_79 must contain 79 people")
        for dataset_id in ("Z2_SEG_EXT1", "Z2_SEG_EXT2", "EXT_MANUAL_50"):
            for key in SELECTION_OPERATIONS.values():
                if self.datasets[dataset_id].get(key, False):
                    raise ProtocolViolation(f"External dataset enables {key}: {dataset_id}")

    def assert_allowed(self, dataset_id: str, operation: str) -> None:
        key = SELECTION_OPERATIONS.get(operation)
        if key is None:
            raise ValueError(f"Unknown selection operation: {operation}")
        if not self.datasets[dataset_id].get(key, False):
            raise ProtocolViolation(f"{operation} is forbidden for {dataset_id}")


def validate_overlap_file(path: Path) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    record = payload["overlaps"]["LS_SEG_200__LS_CLIN_79"]
    if record.get("overlap_status") != "verified" or record.get("patient_overlap") != 0:
        raise ProtocolViolation("LS_SEG_200 vs LS_CLIN_79 overlap truth must be verified zero")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
