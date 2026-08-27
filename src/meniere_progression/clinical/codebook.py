from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..exceptions import ProtocolBlocker


def load_codebook(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def assert_codebook_signed(payload: dict[str, Any]) -> None:
    if payload.get("status") != "signed":
        raise ProtocolBlocker("Clinical codebook is not signed")
