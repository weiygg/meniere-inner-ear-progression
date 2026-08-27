from __future__ import annotations

import pytest

from meniere_progression.reporting.manifests import REQUIRED_MANIFEST_FIELDS, validate_run_manifest


def test_run_manifest_complete() -> None:
    payload = {field: {} for field in REQUIRED_MANIFEST_FIELDS}
    validate_run_manifest(payload)
    payload.pop("git_commit")
    with pytest.raises(ValueError):
        validate_run_manifest(payload)
