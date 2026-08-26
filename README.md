# Ménière disease progression with P-EBM

This code-only repository is being rebuilt as a reproducible study of
multimodal Ménière disease progression using the Parsimonious Event-Based Model
(P-EBM). Cross-sectional P-EBM results are pseudo-temporal orderings; they are not
longitudinal transition rates, causal sequences, or a natural-history model.

## Current status: audit checkpoint

Repository and protected-data auditing was completed on 2026-08-24 on branch
`codex/restructure-pebm-v1`. No final clinical P-EBM was run. The project is paused
before cohort, index-ear, variable-role, and event-state adjudication.

- [Repository audit](docs/REPO_AUDIT.md)
- [Version map](docs/VERSION_MAP.md)
- [Decision log](docs/DECISION_LOG.md)
- [Open issues](docs/OPEN_ISSUES.md)
- [Checkpoint report](docs/CHECKPOINT_01_AUDIT.md)
- [Aggregate hearing-rule sensitivity audit](reports/01_hearing_ear_rule_sensitivity.md)
- [Workspace cleanup log](docs/CLEANUP_LOG.md)
- [Clinical feature data dictionary](docs/DATA_DICTIONARY.md)
- [Protected-data policy](data/README.md)
- [External semicircular-canal Dice re-audit](docs/EXTERNAL_DICE_REAUDIT_20260826.md)
- [ChatGPT review guide](docs/CHATGPT_REVIEW_GUIDE.md)

The hearing-rule audit is exploratory: it does not freeze the affected-ear
definition and does not run P-EBM.

The imaging split is now frozen in `configs/center_split.yaml`: Lishui is the
primary development center, while the two Zhejiang Second Hospital batch families
are separate external-validation strata. External data are test-only and must not
influence model or threshold selection.

The reproducible, PHI-safe audit command is:

```powershell
python scripts\audit_repository.py
```

It writes metadata-only manifests under `data/manifests/`. It does not export
clinical rows or DICOM metadata.

## Safety boundary

The Git repository may contain code, tests, documentation, dependency metadata,
and de-identified aggregate manifests only. The following remain local and are
excluded from Git:

- clinical workbooks and patient-level tables;
- DICOM/NIfTI images and masks;
- model weights and archives;
- patient-level results and local configurations;
- absolute local paths and credentials.

The protected clinical workbook contains direct identifiers. Reading this GitHub
repository is therefore sufficient to review the protocol and implementation, but
not to reconstruct patient-level results without controlled institutional data
access.

## Legacy implementation

The original numbered scripts remain in `src/` for traceability. Several later
scripts and result folders are exploratory or obsolete, even when their directory
name contains `final`. They must not be cited as the final P-EBM analysis. Their
status is recorded in `docs/VERSION_MAP.md`.

The preserved official P-EBM checkout is pinned locally to Parker et al. commit
`ffbe8a969b2947769098f1f4e6099edb32f36b97`. A previous software-only walkthrough
reproduction passed, but this does not validate any clinical model.

## Planned architecture

```text
configs/
data/
  manifests/
  interim/       # local only
  processed/     # local only unless explicitly de-identified and approved
src/meniere_progression/
scripts/
notebooks/
tests/
docs/
results/runs/    # local only
figures/
tables/
reports/
archive/legacy/
```

The next authorized phase starts only after the checkpoint issues are resolved. It
will reconstruct patient/ear/visit/measurement entities and audit PTA definitions
before biomarker eligibility or clinical modelling.
