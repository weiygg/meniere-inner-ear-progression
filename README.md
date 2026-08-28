# Ménière disease progression with P-EBM

This code-only repository is being rebuilt as a reproducible study of
multimodal Ménière disease progression using the Parsimonious Event-Based Model
(P-EBM). Cross-sectional P-EBM results are pseudo-temporal orderings; they are not
longitudinal transition rates, causal sequences, or a natural-history model.

## Current status: Protocol V2 implementation

Protocol V2 implementation started on 2026-08-26 on branch
`codex/protocol-v2-seg-pebm`. The segmentation and clinical-progression tracks are
now explicitly separated. No final clinical P-EBM has been run because the clinical
crosswalk, signed codebook, audiometry timing, and affected/index-ear rules remain
blocked.

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
- [Protocol V2 data and experiment plan](docs/PROTOCOL_V2_DATA_AND_EXPERIMENT_PLAN.md)
- [Protocol V2 implementation status](docs/PROTOCOL_V2_IMPLEMENTATION_STATUS.md)
- [M0 external geometry reliability](reports/M0_GEOMETRY_RELIABILITY.md)
- [M1 five-epoch internal validation pilot](reports/M1_PILOT5_INTERNAL_VALIDATION.md)
- [Complete Protocol V2 execution and code-archive runbook](docs/PROTOCOL_V2_EXECUTION_RUNBOOK.md)
- [Final M1/M2/M3 pilot and exposed-external segmentation results](reports/SEGMENTATION_PILOT5_FINAL_RESULTS.md)

The hearing-rule audit is exploratory: it does not freeze the affected-ear
definition and does not run P-EBM.

The authoritative registry is `configs/dataset_registry.yaml`. `LS_SEG_200` and
`LS_CLIN_79` are different datasets with verified patient overlap of 0; equal local
numbers must never be linked. Lishui segmentation data are development-only, while
the two Zhejiang Second Hospital batch families are external-validation strata from
the same external institution. External data are test-only and must not influence
model, augmentation, loss, threshold, or postprocessing selection.

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

Run the PHI-safe Protocol V2 registry gate with:

```powershell
python scripts\run_pipeline.py registry
```

Clinical phases stop with explicit blockers rather than guessing missing definitions.

The M1 preparation and planner entry points are:

```powershell
python scripts\audit_multiclass_label_overlap.py --manifest <local-m0-manifest.csv> --output data\manifests\m1_multiclass_overlap_audit.json
python scripts\prepare_nnunet_multiclass.py --manifest <local-m0-manifest.csv> --nnunet-raw <local-raw-root> --overlap-policy nearest-exclusive
python scripts\plan_nnunet_protocol_v2.py --nnunet-raw <local-raw-root> --nnunet-preprocessed <local-preprocessed-root> --nnunet-results <local-results-root> --run-dir <local-run-dir>
python scripts\run_nnunet_protocol_v2.py M1 --nnunet-raw <local-raw-root> --nnunet-preprocessed <local-preprocessed-root> --nnunet-results <local-results-root> --run-dir <local-run-dir>
```

Use a pure-ASCII local preprocessing path on Windows because the observed Blosc2
writer failed under a non-ASCII worktree path. Images, labels, weights, logs, and
absolute paths remain ignored.
