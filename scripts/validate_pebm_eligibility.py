from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meniere_progression.pebm.eligibility import evaluate_variable


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a schema-only P-EBM eligibility audit.")
    parser.add_argument("--schema", type=Path, default=ROOT / "data/manifests/clinical_feature_schema.json")
    parser.add_argument("--codebook", type=Path, default=ROOT / "configs/clinical_codebook.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    codebook = yaml.safe_load(args.codebook.read_text(encoding="utf-8"))
    defaults = codebook["variable_role_defaults"]
    role_lookup = {
        variable: role
        for role, variables in defaults.items()
        for variable in variables
    }
    rows = []
    for field in schema["fields"]:
        variable = field["standardized_name"]
        candidate_role = role_lookup.get(variable, field["candidate_role"])
        definition_ready = field["definition_status"] in {"verified", "verified_header_only", "coding_observed"}
        row = {
            "variable": variable,
            "source_center": field["logical_sheet"],
            "unit": field["expected_unit_or_coding"],
            "coding": field["definition_status"],
            "missing_code": "unresolved",
            "abnormal_direction": "unresolved",
            "measurement_window": "unresolved",
            "variable_type": "unresolved",
            "supported_mixture": False,
            "cross_center_compatible": False,
            "collinearity_group": "unresolved",
            "candidate_role": candidate_role,
            "schema_definition_ready": definition_ready,
        }
        rows.append(evaluate_variable(row))
    result = {
        "status": "schema_audit_complete_final_fit_blocked",
        "variable_rows": len(rows),
        "eligible_primary_events": sum(
            row["eligibility_status"] == "eligible" and row["candidate_role"] == "primary_event"
            for row in rows
        ),
        "blocked_rows": sum(row["eligibility_status"] == "blocked" for row in rows),
        "rows": rows,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
