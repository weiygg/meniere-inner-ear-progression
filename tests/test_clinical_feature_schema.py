from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_schema_contains_headers_only_and_excludes_direct_identifiers() -> None:
    payload = json.loads(
        (ROOT / "data/manifests/clinical_feature_schema.json").read_text(encoding="utf-8")
    )
    assert payload["privacy"] == "schema only; zero patient rows or cell values"
    assert len(payload["sheets"]) == 3
    assert len(payload["fields"]) == 73
    assert all(row["github_content"] == "schema_only_no_patient_values" for row in payload["fields"])
    direct = {
        row["original_header"]: row
        for row in payload["fields"]
        if row["original_header"] in {"姓名", "患者姓名", "电话", "病案号", "出生日期"}
    }
    assert direct
    assert all(row["privacy_class"] == "direct_identifier" for row in direct.values())
    assert all(row["candidate_role"] == "exclude_phi" for row in direct.values())
