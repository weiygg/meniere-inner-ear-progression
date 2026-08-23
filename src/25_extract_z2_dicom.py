from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract the five read-only Zhejiang Second Hospital RAR inputs.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archives = sorted(args.input_dir.glob("浙二*.rar"))
    if len(archives) != 5:
        raise RuntimeError(f"Expected 5 Zhejiang Second Hospital archives, found {len(archives)}")
    unrar = Path(r"C:\Program Files\WinRAR\UnRAR.exe")
    if not unrar.exists():
        raise RuntimeError(f"WinRAR extractor not found: {unrar}")
    extraction_rows = []
    for archive in archives:
        print(f"EXTRACT_START {archive.name}", flush=True)
        completed = subprocess.run(
            [str(unrar), "x", "-o+", "-idq", str(archive.resolve()), str(args.output_dir.resolve()) + "\\"],
            check=False,
        )
        extraction_rows.append({"archive": archive.name, "exit_code": completed.returncode})
        print(f"EXTRACT_DONE {archive.name} exit_code={completed.returncode}", flush=True)
    dicom_count = sum(1 for _ in args.output_dir.rglob("*.dcm"))
    study_count = sum(1 for item in args.output_dir.iterdir() if item.is_dir())
    (args.output_dir.parent / "extraction_audit.json").write_text(
        json.dumps(
            {
                "archives": extraction_rows,
                "study_directories": study_count,
                "dicom_files": dicom_count,
                "all_archives_intact": all(row["exit_code"] == 0 for row in extraction_rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"COMPLETE studies={study_count} dicom_files={dicom_count}", flush=True)


if __name__ == "__main__":
    main()
