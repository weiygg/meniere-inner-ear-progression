# Repository audit

Audit date: 2026-08-24 (Asia/Shanghai)
Audit branch: `codex/restructure-pebm-v1`
Baseline branch and commit: `main` at `625bb13b6afb7221d07d39c25d31e740b4811657`

## Audit scope and safety

The checkout combines a small Git code repository with a large, ignored local
medical-data workspace. Raw inputs were read-only. No clinical row, DICOM header,
image, mask, weight, or patient-level output was copied into Git. Protected folders
were inventoried at aggregate level; Git content and non-subject-level legacy
artifacts were inventoried file by file with SHA-256 hashes.

Complete metadata inventories are in `data/manifests/`:

- `repository_file_inventory.csv`: all 86 files tracked at baseline;
- `local_storage_summary.csv`: aggregate local storage by top-level path;
- `raw_data_manifest.csv`: protected source metadata and hashes;
- `legacy_analysis_hashes.csv`: 445 code/data/document artifacts;
- `duplicate_file_groups.csv`: 32 exact duplicate groups;
- `near_duplicate_candidates.csv`: 75 review candidates;
- `clinical_workbook_audit.json`: schema and aggregate-only workbook audit.

## Git baseline

- Initial branch: `main`.
- Initial commit: `625bb13b6afb7221d07d39c25d31e740b4811657` (`Initial code-only research repository`).
- Initial status: clean and synchronized with `origin/main`.
- Branches before this work: local `main` and `origin/main` only.
- Tags: none.
- History: one commit only; ignored local analyses have no Git commit provenance.
- Remote: private repository `weiygg/meniere-inner-ear-progression`.
- New working branch: `codex/restructure-pebm-v1`.

## Repository and local workspace size

The baseline Git repository contained 86 tracked files and approximately 0.71 MB.
The protected/local research workspace contained approximately 29,735 relevant
files and 39.22 GB after excluding Git internals, virtual environments, agent state,
and caches.

| Local area | Files | Bytes | Audit interpretation |
|---|---:|---:|---|
| `data/` | 8,662 before audit manifests | 13,998,965,694 | raw/extracted protected imaging and archives |
| `results_md_progression/` | 11,325 | 9,029,828,038 | mixed intermediate, legacy final, models, masks, and reports |
| `seg4/` | 2,103 | 4,324,697,664 | later segmentation batch; authority not established |
| `seg3/` | 1,807 | 3,909,203,308 | segmentation batch and geometry outputs |
| `analysis_out/` | 2,979 | 3,894,918,606 | earlier modelling/segmentation outputs |
| `xjj内耳分割2/` | 1,885 | 2,786,701,273 | earlier segmentation/geometry batch |
| `xjj内耳分割/` | 835 | 1,265,950,246 | earliest identified segmentation/geometry batch |

Dominant local file types were 13,534 DICOM files, 9,521 `.gz` files, 4,619
`.npz` files, 1,367 PNG files, 209 CSV files, 15 Excel workbooks, 15 Word
documents, 12 PDFs, two notebooks, and one PowerPoint file. No R or R Markdown
analysis was found.

## Tracked code tree

The tracked tree is flat and legacy-oriented:

- `src/`: 57 Python files, mostly numbered `00_` through `53_`;
- `tests/`: eight tracked test files;
- top level: 18 analysis/training scripts plus configuration, environment files,
  README, and Git ignore rules;
- `config/`: one de-identified example YAML.

There was no `src/meniere_progression/` package, no `scripts/run_pipeline.py`, no
run-manifest implementation, and no `configs/analysis.yaml`. At least 12 numbered
phase files are two-line stubs: phases 06, 07, 08, 10, 12, and 14-21. Existing
`run_all.py` covers the earlier audit/reproduction workflow, not the requested final
pipeline.

## Protected datasets

### Clinical workbook

`MD患者评估20260713.xlsx` is the only root-level clinical workbook and has SHA-256
`5a25d6746a31fc19c9f3c7c76b0ca16d1e96afd3a0d3c96c2188c377c9aebb72`.
It is a restricted mixed-level source with direct identifiers and cannot be added to
Git.

| Sheet | Rows after header | Columns | Apparent level | Direct-identifier fields |
|---|---:|---:|---|---|
| 丽水 | 158 | 13 | two ear rows for each of 79 people | name |
| 浙二MD | 112 | 37 | patient/questionnaire/date table | name, telephone, record number, birth date, scan date |
| 浙二 | 206 | 23 | 103 paired-ear visits; prior audit identified 100 baseline people and 3 follow-ups | name, record number |

The source-prefixed baseline denominator from the prior live audit is 179 people
(79 Lishui plus 100 Zhejiang Second Hospital). This is not yet the final P-EBM
cohort.

### PTA audit

- Both ear-level sheets contain 0.5, 1, 2, and 3 kHz columns.
- Neither ear-level sheet contains a 4 kHz column.
- Lishui has 112 complete four-frequency rows; all 112 PTA formulas average exactly
  those four columns.
- Zhejiang Second Hospital has 194 complete four-frequency rows; 192 use the same
  four-column formula and two store numeric PTA values that exactly match the
  four-frequency recomputation.
- The workbook has no saved formula cache for most PTA cells. Downstream code must
  recompute PTA from source frequencies and must not assume `data_only=True` returns
  a value.
- The available columns permit calculation of a separately named
  `PTA_AAOHNS_0.5_1_2_3k`; they do not permit calculation of
  `PTA_study_0.5_1_2_4k` from this workbook.
- Units, masking levels, examination timing, and the provenance of the existing
  AAO-HNS stage values remain unconfirmed. No final stage was reconstructed.

### Imaging and mask archives

- Lishui segmentation archive: 200 subjects/400 ears in the prior read-only archive
  audit; structure names include `Chochlear`, `Cholear`, `ELS`, `HSC`, `PSC`, `SSC`,
  `TV`, and `Vestibular`.
- Zhejiang Second Hospital imaging archives: 94 imaging studies and approximately 91
  unique archived patient keys in the prior audit; archives contain DICOM rather
  than original manual segmentation for the P-EBM cohort.
- Independent external manual-mask archives: two 25-case subsets, 50 cases/100 ears
  in total, with SSC/HSC/PSC masks. These support the frozen segmentation validation
  only; they are not a clinical P-EBM dataset.

All archive hashes are in `raw_data_manifest.csv`. Extracted folders are derived
copies until archive/extraction equivalence is verified.

## Documents, papers, and notebooks

- `pebm-neuroimage.pdf`: 17-page Parker et al. P-EBM paper; first page rendered and
  visually readable.
- `results_md_progression/intermediate/paper/supplementary_file_1.pdf`: eight pages.
- `results_md_progression/intermediate/paper/supplementary_file_2.pdf`: 24 pages.
- `半规管空间结构人工智能MD.pdf`: 12-page semicircular-canal/Ménière paper; first
  page rendered and visually readable.
- Two identical official P-EBM walkthrough notebooks are present: the notebook and
  its checkpoint copy.
- Fifteen Word reports and one progress-deck PPTX are local legacy outputs.

Repository-wide filename and content searches found P-EBM/EBM, PTA, hydrops,
ES/ED, Meniere/Ménière, and Table 1-3 references. No file or content match was found
for `Manuscript PCCT_MD` or `codex_pcct_md_reanalysis.py`. No standalone historical
`Table 1-3` research package was identified; the matches were table references in
PDFs.

## Previous P-EBM analyses

1. **Official software reproduction (2026-07-17).** The local upstream checkout is
   at Parker et al. commit `ffbe8a969b2947769098f1f4e6099edb32f36b97`. Serial and
   simultaneous synthetic walkthroughs previously passed. This is software evidence
   only.
2. **Lishui development, Z2 transport pilot (2026-07-31).** Ear-level analysis used
   the sole staged ear as the affected-ear proxy and its paired ear as a reference.
   It was later declared obsolete by the local locked-cohort document.
3. **Z2 development, Lishui transport pilot (2026-08-01).** Reversed centre roles but
   retained the same affected-ear proxy. It was also explicitly declared obsolete.
4. **Corrected patient-level specification (2026-08-01).** Defined 96 confirmed Z2
   baseline patients and excluded stage from P-EBM inputs. This is a cohort proposal,
   not proof that all mappings and variable definitions are resolved.
5. **Patient-level P-EBM pilot (2026-08-01).** Used worse-ear aggregation for hydrops
   and PTA and treated DHI-T, THI, ear fullness, and VADL as irreversible events.
   Those choices conflict with the current requested event-eligibility rules. This
   result is exploratory/legacy and is not the final model.

## Duplicate and obsolete-output audit

There are 32 exact duplicate groups and 75 normalized-name near-duplicate groups.
Important examples are:

- four identical segmentation `说明.docx` files across different batches;
- identical final/intermediate copies of model metrics, manifests, and training
  histories;
- identical official P-EBM notebook/checkpoint copies;
- two identical header-only unilateral cohort CSVs;
- multiple same-named model reports with different hashes and dates.

No duplicate was deleted or moved. Identical content does not establish which
version is scientifically authoritative. The file-level evidence is in the duplicate
manifests.

## Canonical-source assessment

| Domain | Most defensible current candidate | Status |
|---|---|---|
| Clinical observations | protected root clinical workbook | sole consolidated source, but manual provenance/coding/linkage need confirmation |
| Lishui original segmentation | Lishui RAR archive | source candidate; structure/batch authority unresolved |
| Z2 original imaging | five Z2 RAR archives | source candidate; archive-to-clinical mapping incomplete |
| External canal manual masks | two external validation RAR archives | canonical only for the frozen SSC/HSC/PSC validation |
| P-EBM implementation | Parker et al. commit `ffbe8a9` | pinned reference implementation; local bytecode cache is untracked |
| Clinical P-EBM result | none | all current clinical results remain legacy/exploratory |

## Changes made at this checkpoint

- Created the requested working branch.
- Added a reproducible PHI-safe audit script and metadata manifests.
- Added documentation and target-directory scaffolding.
- Tightened `.gitignore` so `data/` remains protected while metadata manifests can
  be tracked.
- Did not overwrite raw data, delete files, move legacy material, or run a final
  clinical model.

## Post-checkpoint study-team clarification

The study team confirmed that an affected ear is determined by abnormal clinical
hearing features. AAO-HNS-stage presence is not the affected-ear rule. The earlier
150 resolved / 29 unresolved proxy counts therefore remain historical audit counts
and must be recomputed after the exact hearing features, thresholds, and bilateral
handling rule are frozen.

## Post-checkpoint center and version reorganization

On 2026-08-24, the study team froze Lishui as the primary development center and
the two Zhejiang Second Hospital batch families as external validation 1 and 2.
Protected sources were moved under `data/centers/`; the clinical workbook was moved
under `data/clinical/`. The two external groups remain separate test-only strata
and are not claimed to be independent hospitals.

Three older segmentation trees were moved intact to
`archive/legacy/segmentation_versions/`. Three explicitly rejected clinical P-EBM
outputs and three aborted/invalid/empty validation intermediates were deleted.
Two root-level validation archives were deleted only after byte-identical SHA-256
matches were verified at their retained protected destinations. See
`docs/CLEANUP_LOG.md` for the bounded deletion list and recoverability notes.
