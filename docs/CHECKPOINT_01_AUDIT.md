# Checkpoint 01 - repository and data audit

## Completed

- Recorded clean `main@625bb13`, branches, tags, history, and remotes.
- Created `codex/restructure-pebm-v1`.
- Inventoried 86 tracked files and the 39.22-GB protected/local workspace.
- Hashed protected source archives and 445 legacy analysis artifacts.
- Audited the clinical workbook without exporting patient rows.
- Verified the available 0.5/1/2/3-kHz PTA formulas and absence of 4-kHz data.
- Located and classified prior P-EBM runs, official code, papers, duplicates, and
  stubs.
- Added PHI-safe manifests, documentation, and repository scaffolding.

## Blocking scientific decisions

1. canonical clinical source and cross-sheet linkage;
2. exact hearing features/thresholds defining the affected ear, plus bilateral and
   first affected/index-ear handling;
3. clinical variable codebook and normal/abnormal states;
4. authoritative segmentation batch and label map;
5. final progression-biomarker eligibility, especially ordinal hydrops and
   non-monotonic symptoms.

## Current interpretation

No existing clinical P-EBM output is suitable as the final analysis. Two ear-level
pilots are explicitly obsolete. The later patient-level pilot uses event definitions
that conflict with the present protocol. The official software reproduction and the
independent frozen segmentation validation remain useful but answer different
questions.

## Stop gate

The task is intentionally paused here. No processed master tables, event mixtures,
clinical stages, bootstrap, MCMC, figures, or manuscript claims were regenerated.
Proceed only after the requested study-team inputs in `docs/OPEN_ISSUES.md` are
provided or formally adjudicated.
