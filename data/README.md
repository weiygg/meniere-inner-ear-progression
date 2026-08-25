# Protected data boundary

This directory contains local protected clinical/imaging data and is not a Git data
repository. Raw workbooks, DICOM/NIfTI images, masks, archives, model weights, and
patient-level outputs must remain outside Git.

## Frozen local center layout

```text
data/
  clinical/
    MD患者评估20260713.xlsx
  centers/
    primary_lishui/
      raw_archives/
      extracted/
    external_validation_1/
      raw_archives/
      extracted/
      reference_masks/
    external_validation_2/
      raw_archives/
      extracted/
      reference_masks/
  manifests/
```

- `primary_lishui` is the only model-development source and may be split by patient
  for training/internal validation.
- `external_validation_1` contains `浙二1-1` and `浙二1-2`.
- `external_validation_2` contains `浙二2-1`, `浙二2-2`, and `浙二2例新`.
- Both external groups are frozen test-only strata. They must not influence model,
  feature, preprocessing, harmonization, postprocessing, or threshold selection.
- The two external groups originate from Zhejiang Second Hospital and must not be
  described as two independent hospitals without additional provenance evidence.

The pipeline should locate protected inputs through an environment variable such as
`INNER_EAR_DATA_DIR` or a local ignored configuration file. Raw inputs are read-only.

Only the following de-identified metadata are tracked:

- `manifests/raw_data_manifest.csv`;
- `manifests/repository_file_inventory.csv`;
- `manifests/local_storage_summary.csv`;
- `manifests/legacy_analysis_hashes.csv`;
- duplicate/near-duplicate manifests;
- aggregate workbook audit JSON.
- aggregate center-split and workspace-reorganization manifests.

Future `data/interim/` and `data/processed/` products remain local by default. A file
may be committed only after an explicit disclosure-control review confirms that it
contains no patient-level values, direct identifiers, dates, source IDs, paths, or
small-cell re-identification risk.
