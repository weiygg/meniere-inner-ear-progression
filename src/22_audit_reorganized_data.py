from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from mdp_utils import base_visit_id, load_config, read_ear_records, setup_logger, sha256


MASK_FILE = re.compile(
    r"(?P<subject>\d+)(?P<side>[LR])_(?P<structure>[^/\\]+)\.nii(?:\.gz)?$",
    re.IGNORECASE,
)
SUBJECT_DIR = re.compile(r"(?:^|/)(?:sub)?(?P<subject>\d+)(?:/|$)", re.IGNORECASE)


def archive_listing(path: Path) -> list[str]:
    completed = subprocess.run(
        ["tar", "-tf", str(path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    text = completed.stdout.decode("utf-8", errors="replace")
    return [line.strip().replace("\\", "/") for line in text.splitlines() if line.strip()]


def clinical_ids(records: list[dict], site: str) -> set[str]:
    return {
        str(base_visit_id(record["source_subject_id"]))
        for record in records
        if record["source_site"] == site and not record["is_followup"]
    }


def leading_numeric_key(value: str) -> str | None:
    match = re.match(r"^(\d+)", value.strip())
    return f"{int(match.group(1)):03d}" if match else None


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit reorganized LS/Z2 archives without altering them")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    _, paths = load_config(args.config)
    data_dir = args.data_dir.resolve()
    output = paths.output_root / "05_reorganized_data_audit"
    output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("reorganized_data", paths.logs / "22_audit_reorganized_data.log")

    records = read_ear_records(paths.clinical_table)
    ls_clinical = clinical_ids(records, "LS")
    z2_clinical = clinical_ids(records, "Z2")

    archive_rows: list[list[object]] = []
    ls_subjects: set[str] = set()
    ls_ears: set[str] = set()
    ls_structures: dict[str, set[str]] = defaultdict(set)
    z2_subjects: set[str] = set()
    nonnumeric_z2_subjects: set[str] = set()
    z2_studies: set[str] = set()

    for archive in sorted(data_dir.glob("*.rar")):
        listing = archive_listing(archive)
        suffixes = Counter()
        top_dirs: set[str] = set()
        archive_subjects: set[str] = set()
        archive_ears: set[str] = set()
        is_ls = "丽水" in archive.name

        for entry in listing:
            top = entry.split("/", 1)[0]
            if top:
                top_dirs.add(top)
            suffix = ".nii.gz" if entry.lower().endswith(".nii.gz") else Path(entry).suffix.lower() or "[none]"
            suffixes[suffix] += 1
            if is_ls:
                match = MASK_FILE.search(entry)
                if match:
                    subject = f"{int(match.group('subject')):03d}"
                    side = match.group("side").upper()
                    structure = match.group("structure")
                    archive_subjects.add(subject)
                    archive_ears.add(f"{subject}-{side}")
                    ls_structures[f"{subject}-{side}"].add(structure)
                else:
                    subject_match = SUBJECT_DIR.search(entry)
                    if subject_match:
                        archive_subjects.add(f"{int(subject_match.group('subject')):03d}")
                ls_subjects.update(archive_subjects)
                ls_ears.update(archive_ears)
            else:
                if entry.lower().endswith(".dcm"):
                    token = top.strip()
                    z2_studies.add(token)
                    if token.isdigit():
                        normalized = f"{int(token):03d}"
                        archive_subjects.add(normalized)
                        z2_subjects.add(normalized)
                    elif token:
                        nonnumeric_z2_subjects.add(token)

        archive_rows.append(
            [
                archive.name,
                "LS_training" if is_ls else "Z2_external",
                len(listing),
                len(archive_subjects) if is_ls else len(top_dirs),
                len(archive_ears) if is_ls else "",
                json.dumps(dict(suffixes.most_common()), ensure_ascii=False),
                archive.stat().st_size,
                sha256(archive),
            ]
        )

    ls_overlap = ls_subjects & ls_clinical
    z2_clinical_keys = {key for value in z2_clinical if (key := leading_numeric_key(value))}
    z2_archive_patient_keys = {
        key
        for token in z2_studies
        if (key := leading_numeric_key(str(base_visit_id(token))))
    }
    z2_unmapped_patient_tokens = {
        token for token in nonnumeric_z2_subjects if not leading_numeric_key(token)
    }
    z2_overlap = z2_archive_patient_keys & z2_clinical_keys
    structure_counts = Counter(len(value) for value in ls_structures.values())
    expected_structures = sorted({item for values in ls_structures.values() for item in values})
    structure_occurrences = Counter(item for values in ls_structures.values() for item in values)
    core_canals = {"HSC", "PSC", "SSC"}
    incomplete_ears = sorted(
        (ear, ";".join(sorted(core_canals - {item.upper() for item in structures})))
        for ear, structures in ls_structures.items()
        if not core_canals.issubset({item.upper() for item in structures})
    )

    write_csv(
        output / "archive_inventory.csv",
        ["archive", "declared_role", "entry_count", "subject_count", "ear_count", "extension_counts_json", "source_bytes", "sha256"],
        archive_rows,
    )
    write_csv(
        output / "input_manifest.csv",
        ["input", "source_bytes", "sha256"],
        [
            [paths.clinical_table.name, paths.clinical_table.stat().st_size, sha256(paths.clinical_table)],
            *[[row[0], row[-2], row[-1]] for row in archive_rows],
        ],
    )
    write_csv(
        output / "clinical_archive_linkage.csv",
        ["site", "clinical_baseline_subjects", "archive_subjects", "matched_subjects", "clinical_without_archive", "archive_without_clinical"],
        [
            [
                "LS",
                len(ls_clinical),
                len(ls_subjects),
                len(ls_overlap),
                ";".join(sorted(ls_clinical - ls_subjects)),
                ";".join(sorted(ls_subjects - ls_clinical)),
            ],
            [
                "Z2",
                len(z2_clinical),
                len(z2_archive_patient_keys) + len(z2_unmapped_patient_tokens),
                len(z2_overlap),
                ";".join(sorted(z2_clinical_keys - z2_archive_patient_keys)),
                ";".join(sorted(z2_archive_patient_keys - z2_clinical_keys))
                + ((";" if z2_archive_patient_keys - z2_clinical_keys else "") + ";".join(sorted(token for token in nonnumeric_z2_subjects if not leading_numeric_key(token)))),
            ],
        ],
    )
    write_csv(
        output / "ls_incomplete_structure_ears.csv",
        ["ear_id", "missing_structures_relative_to_archive_union"],
        [[ear, missing] for ear, missing in incomplete_ears],
    )

    summary = {
        "declared_split": {
            "training_site": "LS",
            "training_imaging_ears_expected": 400,
            "external_site": "Z2",
            "clinical_workbook_internal_sheet_index": 1,
            "clinical_workbook_external_sheet_index": 3,
        },
        "observed": {
            "ls_archive_subjects": len(ls_subjects),
            "ls_archive_ears": len(ls_ears),
            "ls_clinical_baseline_subjects": len(ls_clinical),
            "ls_clinical_subjects_found_in_archive": len(ls_overlap),
            "z2_archive_imaging_studies": len(z2_studies),
            "z2_archive_unique_patient_keys": len(z2_archive_patient_keys) + len(z2_unmapped_patient_tokens),
            "z2_archive_numeric_patient_keys": len(z2_archive_patient_keys),
            "z2_archive_unmapped_nonnumeric_patient_tokens": len(z2_unmapped_patient_tokens),
            "z2_clinical_baseline_subjects": len(z2_clinical),
            "z2_numeric_subjects_found_in_clinical_sheet": len(z2_overlap),
            "ls_structure_names": expected_structures,
            "ls_structure_ear_occurrences": dict(sorted(structure_occurrences.items())),
            "ls_ear_structure_count_distribution": dict(sorted(structure_counts.items())),
            "ls_ears_missing_one_or_more_semicircular_canal_masks": len(incomplete_ears),
        },
        "interpretation": [
            "The 400-ear denominator describes the LS imaging/segmentation training archive.",
            "Clinical P-EBM denominators are smaller because only workbook rows with the required clinical biomarkers can enter that model.",
            "Z2 archives contain DICOM images rather than manual segmentation masks; external segmentation Dice cannot be computed without reference masks.",
        ],
    }
    (output / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "LS archive subjects=%d ears=%d clinical overlap=%d; Z2 archive subjects=%d numeric clinical overlap=%d",
        len(ls_subjects),
        len(ls_ears),
        len(ls_overlap),
        len(z2_archive_patient_keys) + len(z2_unmapped_patient_tokens),
        len(z2_overlap),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
