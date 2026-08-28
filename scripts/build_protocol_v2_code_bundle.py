from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


ALLOWED_SUFFIXES = {
    ".ini",
    ".json",
    ".md",
    ".mjs",
    ".ps1",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
ALLOWED_SUFFIXLESS = {".gitignore", "LICENSE"}
ALLOWED_PREFIXES = (
    "archive/legacy/",
    "config/",
    "configs/",
    "data/manifests/",
    "docs/",
    "figures/",
    "notebooks/",
    "reports/",
    "scripts/",
    "src/",
    "tables/",
    "tests/",
)
PROTECTED_SUFFIXES = {
    ".7z",
    ".bmp",
    ".ckpt",
    ".csv",
    ".dcm",
    ".doc",
    ".docx",
    ".joblib",
    ".jpeg",
    ".jpg",
    ".mha",
    ".mhd",
    ".nii",
    ".npy",
    ".npz",
    ".nrrd",
    ".pdf",
    ".pickle",
    ".pkl",
    ".png",
    ".ppt",
    ".pptx",
    ".pt",
    ".pth",
    ".rar",
    ".tif",
    ".tiff",
    ".webp",
    ".xls",
    ".xlsx",
    ".zip",
}
PROTECTED_PARTS = {
    ".git",
    ".venv",
    "analysis_out",
    "results",
    "results_md_progression",
    "seg3",
    "seg4",
    "xjj内耳分割",
    "xjj内耳分割2",
}
MAX_MEMBER_BYTES = 5 * 1024 * 1024
SENSITIVE_PATTERNS = {
    "absolute_windows_user_path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\bgh[oprsu]_[A-Za-z0-9_]{20,}\b"),
}
DEPENDENCIES = (
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "scikit-learn",
    "scikit-image",
    "nibabel",
    "SimpleITK",
    "openpyxl",
    "PyYAML",
    "pytest",
    "torch",
    "nnunetv2",
    "monai",
)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, encoding="utf-8"
    ).strip()


def is_allowed_member(relative: str) -> tuple[bool, str]:
    normalized = relative.replace("\\", "/")
    path = PurePosixPath(normalized)
    lower = normalized.lower()
    suffix = path.suffix.lower()
    if lower.endswith(".nii.gz") or suffix in PROTECTED_SUFFIXES:
        return False, "protected_file_type"
    if any(part in PROTECTED_PARTS for part in path.parts):
        return False, "protected_path"
    in_allowed_tree = any(normalized.startswith(prefix) for prefix in ALLOWED_PREFIXES)
    is_allowed_root = len(path.parts) == 1 and (
        suffix in ALLOWED_SUFFIXES or path.name in ALLOWED_SUFFIXLESS
    )
    if not (in_allowed_tree or is_allowed_root):
        return False, "outside_code_allowlist"
    if suffix not in ALLOWED_SUFFIXES and path.name not in ALLOWED_SUFFIXLESS:
        return False, "non_text_or_unapproved_suffix"
    return True, "allowed"


def tracked_code_files(root: Path) -> tuple[list[str], list[dict[str, str]]]:
    tracked = git_output(root, "ls-files").splitlines()
    included: list[str] = []
    excluded: list[dict[str, str]] = []
    for relative in sorted(path for path in tracked if path):
        allowed, reason = is_allowed_member(relative)
        if allowed:
            included.append(relative)
        else:
            excluded.append({"path": relative, "reason": reason})
    return included, excluded


def installed_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def validate_member(root: Path, relative: str) -> tuple[bytes, dict[str, object]]:
    path = root / relative
    content = path.read_bytes()
    if len(content) > MAX_MEMBER_BYTES:
        raise ValueError(f"Refusing member larger than 5 MiB: {relative}")
    text = content.decode("utf-8", errors="replace")
    for name, pattern in SENSITIVE_PATTERNS.items():
        if pattern.search(text):
            raise ValueError(f"Refusing {relative}: detected {name}")
    return content, {
        "path": relative.replace("\\", "/"),
        "bytes": len(content),
        "sha256": sha256_bytes(content),
    }


def build_bundle(root: Path, output_dir: Path) -> tuple[Path, Path, dict[str, object]]:
    root = root.resolve()
    output_dir = output_dir.resolve()
    if git_output(root, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Tracked files are dirty; commit them before creating a code snapshot")

    commit = git_output(root, "rev-parse", "HEAD")
    short_commit = commit[:12]
    branch = git_output(root, "branch", "--show-current") or "detached"
    files, excluded = tracked_code_files(root)
    if not files:
        raise RuntimeError("No tracked code files matched the allowlist")

    inventory: list[dict[str, object]] = []
    contents: dict[str, bytes] = {}
    for relative in files:
        content, record = validate_member(root, relative)
        contents[relative] = content
        inventory.append(record)

    created_utc = datetime.now(timezone.utc).isoformat()
    metadata: dict[str, object] = {
        "bundle_format_version": 1,
        "project": "meniere-inner-ear-progression Protocol V2",
        "created_utc": created_utc,
        "source_commit": commit,
        "source_branch": branch,
        "archive_scope": "tracked code, configuration, tests, documentation, and text-only aggregate manifests",
        "privacy_boundary": {
            "patient_tables": False,
            "dicom_or_nifti": False,
            "masks": False,
            "model_weights": False,
            "patient_level_results": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "installed_versions": installed_versions(),
        },
        "included_file_count": len(inventory),
        "excluded_tracked_files": excluded,
        "inventory": inventory,
    }
    metadata_bytes = (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"meniere_protocol_v2_code_{short_commit}.zip"
    bundle_root = "meniere_protocol_v2_code"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for relative in sorted(contents):
            handle.writestr(f"{bundle_root}/{relative}", contents[relative])
        handle.writestr(f"{bundle_root}/BUNDLE_METADATA.json", metadata_bytes)

    sidecar = archive.with_suffix(".sha256.json")
    verification = {
        "archive": archive.name,
        "bytes": archive.stat().st_size,
        "sha256": sha256_file(archive),
        "source_commit": commit,
        "source_branch": branch,
        "included_file_count": len(inventory),
        "created_utc": created_utc,
    }
    sidecar.write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return archive, sidecar, verification


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a PHI-safe, hash-inventoried Protocol V2 code-only ZIP."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    archive, sidecar, verification = build_bundle(args.root, args.output_dir)
    print(
        json.dumps(
            {
                "status": "pass",
                "archive": str(archive),
                "checksum_file": str(sidecar),
                **verification,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
