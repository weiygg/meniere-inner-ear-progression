from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


PROTECTED_SUFFIXES = {
    ".dcm", ".nii", ".nrrd", ".mha", ".mhd", ".npz", ".npy", ".pt", ".pth", ".ckpt",
    ".pkl", ".pickle", ".joblib", ".xls", ".doc", ".docx", ".ppt", ".pptx", ".pdf",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".zip", ".rar", ".7z",
}
PROTECTED_PREFIXES = (
    "results/", "results_md_progression/", "analysis_out/", "seg3/", "seg4/",
    "xjj内耳分割/", "xjj内耳分割2/",
)
ALLOWED_BINARY = {"data/manifests/clinical_feature_schema.xlsx"}
TEXT_SUFFIXES = {".py", ".ps1", ".md", ".json", ".yaml", ".yml", ".toml", ".txt"}
SENSITIVE_PATTERNS = {
    "absolute_windows_user_path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\bgh[oprsu]_[A-Za-z0-9_]{20,}\b"),
}


def candidate_files(root: Path) -> list[str]:
    command = ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
    output = subprocess.check_output(command, cwd=root, text=True, encoding="utf-8")
    return sorted(path for path in output.splitlines() if path)


def audit(root: Path) -> dict[str, object]:
    files = candidate_files(root)
    violations: list[dict[str, str]] = []
    for relative in files:
        normalized = relative.replace("\\", "/")
        path = root / relative
        lower = normalized.lower()
        suffix = path.suffix.lower()
        if lower.endswith(".nii.gz") or (
            suffix in PROTECTED_SUFFIXES and normalized not in ALLOWED_BINARY
        ):
            violations.append({"path": normalized, "reason": "protected_file_type"})
            continue
        if normalized.startswith(PROTECTED_PREFIXES):
            violations.append({"path": normalized, "reason": "protected_output_path"})
            continue
        if path.exists() and path.stat().st_size > 5 * 1024 * 1024:
            violations.append({"path": normalized, "reason": "file_larger_than_5MiB"})
        if suffix in TEXT_SUFFIXES and path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            for name, pattern in SENSITIVE_PATTERNS.items():
                if pattern.search(text):
                    violations.append({"path": normalized, "reason": name})
    return {
        "status": "pass" if not violations else "fail",
        "candidate_file_count": len(files),
        "violations": violations,
        "scope": "tracked_and_unignored_untracked_files",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail before Git publication if protected content is present.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = audit(args.root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
