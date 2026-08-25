# Workspace cleanup and version-management log

Date: 2026-08-24
Status: applied
Machine-readable audit: `data/manifests/workspace_reorganization_20260824.json`

## Final center layout

- Primary development: `data/centers/primary_lishui/`.
- External validation 1: `data/centers/external_validation_1/`.
- External validation 2: `data/centers/external_validation_2/`.
- Protected clinical workbook: `data/clinical/`.

The split is frozen in `configs/center_split.yaml`. External validation 1 and 2
are test-only and cannot participate in preprocessing/model/feature/threshold
selection. Both are Zhejiang Second Hospital strata rather than confirmed separate
hospitals.

## Archived, not deleted

| Old path | New path | Reason |
|---|---|---|
| `xjj内耳分割/` | `archive/legacy/segmentation_versions/xjj内耳分割/` | older mixed segmentation/geometry tree; authority unresolved |
| `xjj内耳分割2/` | `archive/legacy/segmentation_versions/xjj内耳分割2/` | older mixed segmentation/geometry tree; authority unresolved |
| `seg3/` | `archive/legacy/segmentation_versions/seg3/` | superseded by the current `seg4` candidate |

These moves preserve 4,527 files and their content. `seg4/` remains in place as the
current candidate development tree; the protected Lishui archive remains the raw
source candidate.

## Deleted with explicit evidence

| Deleted item | Files | Bytes | Reason / recoverability |
|---|---:|---:|---|
| root duplicate `中心2外部验证1.rar` | 1 | 665,318,145 | exact SHA-256 match retained under external validation 1 |
| root duplicate `中心3外部验证2.rar` | 1 | 644,363,886 | exact SHA-256 match retained under external validation 2 |
| `clinical_pebm_external_validation_20260731/` | 21 | 3,128,716 | obsolete stage-cell affected-ear proxy; reproducible from preserved code/source if ever needed |
| `clinical_pebm_z2_development_20260801/` | 12 | 2,181,027 | reversed-center pilot with the same rejected proxy |
| `patient_level_md_pebm_20260801/` | 8 | 97,956 | worse-ear and symptom-event choices conflict with the current protocol |
| aborted/invalid/empty external-validation intermediates | 21 | 14,249,157 | superseded by `external_manual_validation_20260817/` |

Total deleted: 64 files / 1,329,338,887 bytes. Unique raw source content and the
valid frozen external validation were not deleted. Deleted obsolete analysis
directories are no longer locally recoverable as files, but their generating code,
source inputs, and pre-cleanup file hashes remain available.

## Deliberately retained

- `results_md_progression/final/external_manual_validation_20260817/`;
- current semicircular-canal and six-structure final models and their supporting
  training/intermediate records;
- official P-EBM source/walkthrough reproduction;
- earlier `analysis_out/` experiments because a retained transfer-learning script
  still references a prior checkpoint;
- `study_design_corrected_20260801/` as audit evidence;
- raw archives, extracted DICOM/NIfTI data, manual masks, clinical workbook,
  manuscripts, figures, tables, and model weights.
