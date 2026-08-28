# Protocol V2 implementation status

Date: 2026-08-26

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
| M2 training | equal-budget pilot running | same split and 5-epoch budget; fixed bounded bias-field augmentation; no external labels |
| M3 | executable, not yet trained | fixed M2 augmentation plus 0.1 soft-clDice trainer implemented |
| Current external 50 | locked | aggregate result documented; forbidden as a selection source |
| Blinded inter-reader sampling | script implemented | patient UID output remains local and ignored |
| Geometry reliability | continuous estimates completed | 50 people/100 ears/300 mask pairs; patient bootstrap 1,000; formal pass threshold remains blocked |
| P-EBM eligibility | schema-only validator implemented | final fit remains blocked |
| Final clinical P-EBM | not run | crosswalk, codebook, timing, and affected-ear gates unresolved |
| New confirmatory external cohort | not found | prospective requirement documented |
