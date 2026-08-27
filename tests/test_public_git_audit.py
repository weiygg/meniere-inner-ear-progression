from __future__ import annotations

from pathlib import Path


def test_public_audit_patterns_do_not_match_placeholder_paths() -> None:
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_public_git.py"
    spec = importlib.util.spec_from_file_location("audit_public_git", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    safe = "python script.py --input <local-data-root>"
    assert all(pattern.search(safe) is None for pattern in module.SENSITIVE_PATTERNS.values())
