# Protocol V2 implementation status

Date: 2026-09-02

| Area | Status | Evidence / next gate |
|---|---|---|
| Dataset registry and permissions | implemented | seven Protocol V2 YAML configurations |
| LS segmentation vs clinical identity | resolved | overlap=0 recorded; namespace and no-bare-join tests |
| Package-level identifiers/registry | implemented | `src/meniere_progression/identifiers.py`, `registry.py` |
| Clinical PTA and affected-ear guard | implemented/blocked | 0.5/1/2/3-kHz function; final rule requires signed inputs |
| Phase runner and run manifest | implemented | `scripts/run_pipeline.py` |
| M0 split provenance | completed | 200 people/400 ears; 140/30/30 people in train/validation/internal benchmark; manifest hash recorded |
| Legacy `sub128` HSC filename | corrected/audited | `138R_HSC.nii.gz` renamed to `128R_HSC.nii.gz` after SHA-256 identity check against canonical copy |
| M1 source-label overlap | audited/resolved | 2,565 shared voxels in 195/400 ears; explicit nearest-exclusive-core conversion frozen in D021 |
| M1 dataset and planner | completed | 280 train, 60 validation, 60 isolated benchmark ears; nnU-Net 2.6.2 `3d_fullres`; planner hashes recorded |
| M1 training | equal-budget pilot completed | 5 epochs; 30 people/60 ears; macro Dice 0.7964 (patient-bootstrap 95% CI 0.7882-0.8046); internal validation only |
| M2 training | equal-budget pilot completed/selected | 5 epochs; internal macro Dice 0.7984 (95% CI 0.7909-0.8058); selected among M1-M3 using internal validation only |
| M3 training | equal-budget pilot completed/not selected | 5 epochs; internal macro Dice 0.7957 (95% CI 0.7873-0.8038); fixed M2 augmentation plus 0.1 soft-clDice |
| Current external 50 | locked exposed evaluation completed | selected M2 pooled macro Dice 0.6696 (95% CI 0.6527-0.6867); Center 2 0.6554 and Center 3 0.6837; target 0.78 not reached; forbidden as a selection source |
| Formal E1 fold-0 rapid benchmark | completed/exploratory | 54-epoch checkpoint; internal fold-0 macro Dice 0.8153; exposed-external pooled macro Dice 0.6817 (95% CI 0.6643-0.6987), paired +0.0122 vs M2; target 0.78 not reached; folds 1-4 incomplete |
| Formal E1 predicted-mask features | completed/exploratory | 50 people/100 ears/300 masks; 300 basic-shape rows, 299/300 centerlines passing QC, 298 plane angles, and 450 bilateral-asymmetry rows; patient-level workbook remains local; GitHub artifact is aggregate-only |
| Blinded inter-reader sampling | script implemented | patient UID output remains local and ignored |
| Geometry reliability | continuous estimates completed | 50 people/100 ears/300 mask pairs; patient-clustered bootstrap 5,000; formal pass threshold remains blocked |
| P-EBM eligibility | schema-only audit completed | 73 variable rows audited; 0 eligible primary events and 73 blocked rows; final fit remains blocked |
| Final clinical P-EBM | not run | crosswalk, codebook, timing, and affected-ear gates unresolved |
| New confirmatory external cohort | not found | prospective requirement documented |
