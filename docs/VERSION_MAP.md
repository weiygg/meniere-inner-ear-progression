# Version map

Local filesystem creation times are not reliable scientific provenance and can
change during copying. Modified times and SHA-256 hashes are used as traceability
evidence. Except for the code-only baseline, the entries below are Git-ignored and
have no repository commit.

| Version / path | Modified / commit | Sample size and unit | Main variables or contents | Outputs / relationship | Classification and status |
|---|---|---|---|---|---|
| `data/clinical/MD患者评估20260713.xlsx` | 2026-07-13; SHA-256 `5a25d674...aebb72`; no Git commit | 158 LS ear rows; 112 Z2 patient/questionnaire rows; 206 Z2 ear rows; prior audit: 179 baseline people | demographics, direct identifiers, ear side, CochEH, VestEH, VA, ES/ED, 0.5/1/2/3 kHz, PTA formulas, AAO-HNS stage, symptom scores | moved from repository root into the protected data boundary on 2026-08-24 | protected manually assembled source; canonical candidate only after provenance and linkage confirmation |
| `data/centers/primary_lishui/raw_archives/丽水-xjj内耳分割4.rar` | source hash `55727783...aa04`; no Git commit | 200 people / 400 ears | NIfTI masks for multiple inner-ear structures | frozen as the sole primary development archive | protected raw archive; likely source candidate, authority/label map unresolved |
| `data/centers/external_validation_1/raw_archives/浙二1-1.rar` + `浙二1-2.rar` | hashes `c3c09f65...b3782`, `e9e79e87...bd9f0`; no Git commit | prior audit: 51 imaging studies | DICOM T2 studies | frozen external validation 1; test-only | protected raw imaging source |
| `data/centers/external_validation_2/raw_archives/浙二2-1.rar` + `浙二2-2.rar` + `浙二2例新.rar` | hashes `453d4f29...d10bf`, `01b14daf...e727`, `8cd83254...51afd`; no Git commit | prior audit: 43 imaging studies | DICOM T2 studies | frozen external validation 2; test-only | protected raw imaging source; one prior truncated-study flag remains relevant |
| `archive/legacy/segmentation_versions/xjj内耳分割/` | local modified dates from 2026-02 onward; no Git commit | earlier segmentation batch | masks, volumes, planes, angles | moved from root on 2026-08-24; precedes later batches | legacy derived/source mix; retained, not canonical |
| `archive/legacy/segmentation_versions/xjj内耳分割2/` | local modified dates from 2026-02 to 2026-04; no Git commit | later segmentation/geometry batch | masks, geometry, models | moved from root on 2026-08-24; partially duplicated later | legacy; retained, role not fully determined |
| `archive/legacy/segmentation_versions/seg3/` | local outputs mainly 2026-05; no Git commit | segmentation batch with geometry outputs | SSC/HSC/PSC geometry and volume | moved from root on 2026-08-24; `seg4` remains active candidate | legacy retained version |
| `seg4/` | local outputs through 2026-07; no Git commit | later segmentation batch | development masks and imaging | used by later segmentation code; newer filename is insufficient evidence of authority | candidate development source; confirmation required |
| `results_md_progression/01_data_audit/` | 2026-07-17; audit workbook SHA-256 `b3452b70...e8086`; no Git commit | 179 unique baseline people; 358 baseline ear rows; 3 repeat visits | workbook schema, missingness, patient-ear linkage | first formal data audit; 29 index-ear cases remained unresolved | derived audit; preserved evidence |
| `results_md_progression/03_morphometry/morphometry_features.xlsx` | 2026-07-17; SHA-256 `2b668bd7...fd70`; no Git commit | 4,130 masks; 3,011 canal centerlines | morphology, geometry, QC | repaired first-round extraction output | derived analysis; not yet clinically linked/canonical |
| `results_md_progression/04_pebm/reproduction_report.md` | 2026-07-17; SHA-256 `7bb07924...4e5d`; upstream commit `ffbe8a9` | synthetic serial/simultaneous datasets | official P-EBM event ordering, MCMC, likelihood | confirms software walkthrough only | reproducibility evidence; valid but not a clinical result |
| deleted `final/clinical_pebm_external_validation_20260731/` | 2026-07-31; workbook SHA-256 `069013c2...c377` | LS 56 people/112 ears; Z2 94 people/188 ears | hydrops, PTA or four frequencies; AAO stage-derived affected-ear proxy | LS-fit/Z2-transport pilot; deleted 2026-08-24 after explicit rejection | obsolete derived clinical pilot; hashes/code/source retained for audit/regeneration |
| deleted `final/clinical_pebm_z2_development_20260801/` | 2026-08-01; workbook SHA-256 `8adf8c0f...5473` | Z2 94 people/188 ears; LS 56 people/112 ears | same proxy and biomarker panels with reversed centre roles | superseded previous pilot; deleted 2026-08-24 | obsolete derived clinical pilot; hashes/code/source retained for audit/regeneration |
| `final/study_design_corrected_20260801/cohort_summary.json` | 2026-08-01; SHA-256 `08f24c26...5e5` | 100 Z2 baseline people, proposed 96 confirmed-MD primary patients | patient-level aggregation; AAO stage excluded from P-EBM | corrects the two ear-level proxy pilots | derived cohort specification; provisional until mappings/codebook are adjudicated |
| deleted `final/patient_level_md_pebm_20260801/pebm_summary.json` | 2026-08-01; SHA-256 `551efe67...1643` | 96 Z2 baseline patients | worse-ear hydrops/PTA plus DHI-T, THI, ear fullness, VADL events | conflicting exploratory model deleted 2026-08-24 | obsolete derived analysis; hashes/code/source retained for audit/regeneration |
| `data/centers/external_validation_1/reference_masks/中心2外部验证1.rar` + `data/centers/external_validation_2/reference_masks/中心3外部验证2.rar` | hashes `4c85b08b...853`, `d199edd2...a56`; validated 2026-08-17 | 25 + 25 people; 100 ears; 300 canal masks | SSC/HSC/PSC manual masks | frozen external segmentation validation | protected raw validation sources; canonical for this segmentation endpoint only |
| `final/external_manual_validation_20260817/` | 2026-08-17; no Git commit | 50 people / 100 ears | Dice, IoU, HD95, ASSD, surface Dice, volume error | supersedes the pre-override invalid intermediate run; separate from clinical P-EBM | derived final segmentation validation; preserve, do not reinterpret as P-EBM evidence |
| baseline Git repository | commit `625bb13b6afb7221d07d39c25d31e740b4811657` | 86 code/config/test files | numbered pipeline, segmentation, P-EBM wrappers, tests | first code-only publication | Git source; current history baseline |

## Move and deletion map

The protected data and legacy trees are Git-ignored, so filesystem moves rather
than `git mv` were used. All old/new paths, file counts, byte counts, hashes for
moved archives, and deletion reasons are recorded in
`data/manifests/workspace_reorganization_20260824.json` and summarized in
`docs/CLEANUP_LOG.md`. No tracked historical file was moved or deleted.
