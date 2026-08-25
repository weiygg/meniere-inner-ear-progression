# Clinical workbook data dictionary

This document exports **schema only** from the protected clinical workbook. It contains no patient rows, cell values, names, identifiers, dates, images, or report text.

Candidate roles are provisional. Fields marked `unresolved` require the study codebook and must not be silently recoded or entered into P-EBM.

## Sheet overview

| Source sheet | Logical sheet | Level | Source rows not exported | Columns | Cohort role |
|---|---|---|---:|---:|---|
| 丽水 | Lishui_ear_level | ear-level; paired left/right rows | 158 | 13 | primary development clinical sheet |
| 浙二MD | Z2_patient_questionnaire | patient/questionnaire-level | 112 | 37 | restricted linkage/questionnaire source; crosswalk unresolved |
| 浙二 | Z2_ear_visit_level | ear- and visit-level; paired left/right rows | 206 | 23 | external clinical sheet; baseline and follow-up visits present |

## 主要特征组

- 人口学与协变量：年龄、性别、身高、体重、教育年限、利手。
- MRI/内耳影像：耳蜗积水（CochEH）、前庭积水（VestEH）、VA、ES/ED、半规管及前庭导水管显影。
- 听力学：0.5/1/2/3 kHz听阈、四频PTA、AAO-HNS听力分期、纯音听阈报告、声导抗。
- 前庭与症状负担：前庭试验、DHI各维度、VADL、UCLA-DQ、EEV、耳闷。
- 耳鸣、偏头痛、睡眠及认知：THI、HIT-6、MIDAS、PSQI、MMSE、MoCA、HADS。
- 标识与时间字段：仅用于受控环境中的患者/耳/访视关联，不上传字段值，也不进入模型。

## Privacy boundary

- Direct identifiers (`姓名`, `患者姓名`, `电话`, `病案号`, `出生日期`) are listed only as field names and are excluded from modelling and GitHub data export.
- Dates, source IDs, free text, document references, and examination narratives remain restricted; their values are not exported.
- The uploaded workbook and JSON contain definitions only. They cannot reconstruct the clinical cohort.

## 丽水

| Original header | Standardized name | Group | Meaning | Unit/coding | Privacy | Candidate role | Definition status |
|---|---|---|---|---|---|---|---|
| ID | source_subject_or_visit_id | identifier | Site-local source subject/visit identifier. | source-specific text/number | restricted_identifier | identifier_only | verified_header_only |
| 姓名 | patient_name | direct_identifier | Patient name. | text | direct_identifier | exclude_phi | direct_identifier |
| age | age_years | demographic | Age at the recorded assessment. | years | clinical_metadata_no_values_exported | candidate_covariate | unit_expected_not_signed |
| sex | sex | demographic | Recorded sex. | coding requires source codebook | clinical_metadata_no_values_exported | candidate_covariate | unresolved_coding |
| side | ear_side | ear_identifier | Ear side. | L/R | restricted_identifier | identifier_only | coding_observed |
| CochEH | cochlear_endolymphatic_hydrops | mri_hydrops | Cochlear endolymphatic-hydrops assessment. | ordinal coding; codebook required | clinical_metadata_no_values_exported | candidate_progression_biomarker | unresolved_coding |
| VestEH | vestibular_endolymphatic_hydrops | mri_hydrops | Vestibular endolymphatic-hydrops assessment. | ordinal coding; codebook required | clinical_metadata_no_values_exported | candidate_progression_biomarker | unresolved_coding |
| stage（AAO-HNS） | aao_hns_hearing_stage | clinical_validator | Recorded AAO-HNS hearing stage. | ordinal stage; provenance/timing require confirmation | clinical_metadata_no_values_exported | validator_not_pebm_input | provenance_unresolved |
| 0.5kHZ | hearing_threshold_0_5khz | audiometry | Pure-tone hearing threshold at 0.5 kHz. | dB HL expected; unit not confirmed | clinical_metadata_no_values_exported | candidate_progression_biomarker | unit_unconfirmed |
| 1kHZ | hearing_threshold_1khz | audiometry | Pure-tone hearing threshold at 1 kHz. | dB HL expected; unit not confirmed | clinical_metadata_no_values_exported | candidate_progression_biomarker | unit_unconfirmed |
| 2kHZ | hearing_threshold_2khz | audiometry | Pure-tone hearing threshold at 2 kHz. | dB HL expected; unit not confirmed | clinical_metadata_no_values_exported | candidate_progression_biomarker | unit_unconfirmed |
| 3kHZ | hearing_threshold_3khz | audiometry | Pure-tone hearing threshold at 3 kHz. | dB HL expected; unit not confirmed | clinical_metadata_no_values_exported | candidate_progression_biomarker | unit_unconfirmed |
| PTA | pta_aaohns_0_5_1_2_3khz | audiometry | Four-frequency mean of 0.5, 1, 2, and 3 kHz in the ear-level sheets. | dB HL expected; recompute from frequencies | clinical_metadata_no_values_exported | candidate_progression_biomarker | formula_audited_unit_unconfirmed |

## 浙二MD

| Original header | Standardized name | Group | Meaning | Unit/coding | Privacy | Candidate role | Definition status |
|---|---|---|---|---|---|---|---|
| 患者姓名 | patient_name | direct_identifier | Patient name. | text | direct_identifier | exclude_phi | direct_identifier |
| 电话 | telephone | direct_identifier | Telephone number. | text | direct_identifier | exclude_phi | direct_identifier |
| 编号 | source_patient_number | identifier | Source-local patient number; sheet-to-sheet crosswalk is not confirmed. | source-specific identifier | restricted_identifier | identifier_only | unresolved_crosswalk |
| MD1,SD2,其他3 | diagnosis_group | diagnosis | Source diagnosis-group code. | 1=MD; 2=SD; 3=other per header; exact criteria needed | clinical_metadata_no_values_exported | cohort_eligibility | criteria_unresolved |
| 起病时间 | disease_onset_time | clinical_timing | Reported disease-onset time/date. | date or duration; codebook needed | sensitive_date_or_quasi_identifier | candidate_covariate | unresolved_definition |
| 病案号 | medical_record_number | direct_identifier | Hospital medical-record number. | identifier | direct_identifier | exclude_phi | direct_identifier |
| 性别（男=1，女=2） | sex | demographic | Recorded sex; source header states male=1 and female=2. | 1=male; 2=female | clinical_metadata_no_values_exported | candidate_covariate | coding_in_header |
| 出生日期 | date_of_birth | direct_identifier | Date of birth. | date | direct_identifier | exclude_phi | direct_identifier |
| 扫描日期 | scan_date | sensitive_date | MRI examination date. | date | sensitive_date_or_quasi_identifier | linkage_only | restricted_date |
| 年龄 | age_years | demographic | Age at the recorded assessment. | years | clinical_metadata_no_values_exported | candidate_covariate | unit_expected_not_signed |
| 身高cm | height_cm | anthropometry | Height. | cm | clinical_metadata_no_values_exported | candidate_covariate | unit_in_header |
| 体重kg | weight_kg | anthropometry | Weight. | kg | clinical_metadata_no_values_exported | candidate_covariate | unit_in_header |
| 教育年限y | education_years | demographic | Years of education. | years | clinical_metadata_no_values_exported | candidate_covariate | unit_in_header |
| 利手（右=0，左=1） | handedness | demographic | Handedness. | 0=right; 1=left | clinical_metadata_no_values_exported | candidate_covariate | coding_in_header |
| MMSE | mmse_total | cognitive_scale | Mini-Mental State Examination total score. | score; version/range require codebook | clinical_metadata_no_values_exported | candidate_validator_or_covariate | instrument_details_unconfirmed |
| HADS（≥12 +） | hads_source_score | psychological_scale | Hospital Anxiety and Depression Scale source field; header embeds a >=12 positive rule. | score; subscale/total and threshold provenance require confirmation | clinical_metadata_no_values_exported | candidate_validator | instrument_details_unconfirmed |
| DHI-F | dhi_functional | vertigo_disability | Dizziness Handicap Inventory functional subscore. | score | clinical_metadata_no_values_exported | clinical_validator | instrument_version_unconfirmed |
| DHI-E | dhi_emotional | vertigo_disability | Dizziness Handicap Inventory emotional subscore. | score | clinical_metadata_no_values_exported | clinical_validator | instrument_version_unconfirmed |
| DHI-P | dhi_physical | vertigo_disability | Dizziness Handicap Inventory physical subscore. | score | clinical_metadata_no_values_exported | clinical_validator | instrument_version_unconfirmed |
| DHI-T | dhi_total | vertigo_disability | Dizziness Handicap Inventory total score. | score | clinical_metadata_no_values_exported | clinical_validator_not_irreversible_event | instrument_version_unconfirmed |
| VADL | vadl_total | activities_daily_living | Vestibular Activities of Daily Living source score. | score; version/range require codebook | clinical_metadata_no_values_exported | clinical_validator_not_irreversible_event | instrument_details_unconfirmed |
| UCLA-DQ | ucla_dizziness_questionnaire | vertigo_scale | UCLA Dizziness Questionnaire source score. | score; version/range require codebook | clinical_metadata_no_values_exported | clinical_validator | instrument_details_unconfirmed |
| EEV | eev_source_score | vertigo_scale | Source field labelled EEV; instrument expansion and scoring require confirmation. | score; codebook required | clinical_metadata_no_values_exported | clinical_validator | unresolved_abbreviation |
| MoCA | moca_total | cognitive_scale | Montreal Cognitive Assessment total score. | score; version/language require codebook | clinical_metadata_no_values_exported | candidate_validator_or_covariate | instrument_details_unconfirmed |
| HIT-6 | hit6_total | headache_impact | Six-item Headache Impact Test total score. | score | clinical_metadata_no_values_exported | clinical_validator | instrument_version_unconfirmed |
| MIDAS | midas_total | migraine_disability | Migraine Disability Assessment source score. | score | clinical_metadata_no_values_exported | clinical_validator | instrument_version_unconfirmed |
| PSQI | psqi_total | sleep_quality | Pittsburgh Sleep Quality Index source score. | score | clinical_metadata_no_values_exported | clinical_validator | instrument_version_unconfirmed |
| THI | thi_total | tinnitus_handicap | Tinnitus Handicap Inventory source score. | score | clinical_metadata_no_values_exported | clinical_validator_not_irreversible_event | instrument_version_unconfirmed |
| 耳闷 | aural_fullness | symptom | Aural fullness source field. | binary/ordinal/text coding requires codebook | clinical_metadata_no_values_exported | clinical_validator_not_irreversible_event | unresolved_coding |
| 内耳检查结果 | inner_ear_exam_result | clinical_exam | Inner-ear examination result field. | text/categorical; codebook required | clinical_metadata_no_values_exported | source_review_only | unresolved_coding |
| 半规管显影 | semicircular_canal_visibility | imaging_exam | Semicircular-canal visualization field. | text/categorical; protocol and coding required | clinical_metadata_no_values_exported | candidate_imaging_feature | unresolved_coding |
| 前庭导水管显影 | vestibular_aqueduct_visibility | imaging_exam | Vestibular-aqueduct visualization field. | text/categorical; protocol and coding required | clinical_metadata_no_values_exported | candidate_endotype | unresolved_coding |
| 纯音听阈报告 | pure_tone_audiogram_report | source_document | Pure-tone audiometry report indicator/reference. | text/document reference | potentially_identifiable_content | linkage_or_source_review | unresolved_content |
| 声导抗 | immittance_audiometry | audiology_test | Acoustic immittance/tympanometry source field. | text/categorical; protocol and coding required | clinical_metadata_no_values_exported | candidate_validator | unresolved_coding |
| 前庭试验 | vestibular_testing | vestibular_test | Vestibular-test source field. | test type/result coding required | clinical_metadata_no_values_exported | candidate_progression_biomarker_or_validator | unresolved_coding |
| 其他 | other_notes | free_text | Other clinical notes. | free text | potentially_identifiable_content | exclude_until_adjudicated | unstructured_sensitive_text |
| <blank> | unnamed_column | unclassified | Blank source header; content role is not established. | unknown | clinical_metadata_no_values_exported | exclude | unresolved |

## 浙二

| Original header | Standardized name | Group | Meaning | Unit/coding | Privacy | Candidate role | Definition status |
|---|---|---|---|---|---|---|---|
| ID | source_subject_or_visit_id | identifier | Site-local source subject/visit identifier. | source-specific text/number | restricted_identifier | identifier_only | verified_header_only |
| 病案号 | medical_record_number | direct_identifier | Hospital medical-record number. | identifier | direct_identifier | exclude_phi | direct_identifier |
| 姓名 | patient_name | direct_identifier | Patient name. | text | direct_identifier | exclude_phi | direct_identifier |
| age | age_years | demographic | Age at the recorded assessment. | years | clinical_metadata_no_values_exported | candidate_covariate | unit_expected_not_signed |
| sex | sex | demographic | Recorded sex. | coding requires source codebook | clinical_metadata_no_values_exported | candidate_covariate | unresolved_coding |
| side | ear_side | ear_identifier | Ear side. | L/R | restricted_identifier | identifier_only | coding_observed |
| CochEH | cochlear_endolymphatic_hydrops | mri_hydrops | Cochlear endolymphatic-hydrops assessment. | ordinal coding; codebook required | clinical_metadata_no_values_exported | candidate_progression_biomarker | unresolved_coding |
| VestEH | vestibular_endolymphatic_hydrops | mri_hydrops | Vestibular endolymphatic-hydrops assessment. | ordinal coding; codebook required | clinical_metadata_no_values_exported | candidate_progression_biomarker | unresolved_coding |
| VA | va_source_field | imaging_anatomy | Source field labelled VA; exact anatomical meaning and coding require confirmation. | codebook required | clinical_metadata_no_values_exported | candidate_endotype_or_exclude | unresolved_abbreviation |
| ES/ED | es_ed_source_field | imaging_biomarker | Source field labelled ES/ED; exact definition, direction, and coding require confirmation. | codebook required | clinical_metadata_no_values_exported | candidate_biomarker | unresolved_abbreviation |
| stage（AAO-HNS） | aao_hns_hearing_stage | clinical_validator | Recorded AAO-HNS hearing stage. | ordinal stage; provenance/timing require confirmation | clinical_metadata_no_values_exported | validator_not_pebm_input | provenance_unresolved |
| 0.5kHZ | hearing_threshold_0_5khz | audiometry | Pure-tone hearing threshold at 0.5 kHz. | dB HL expected; unit not confirmed | clinical_metadata_no_values_exported | candidate_progression_biomarker | unit_unconfirmed |
| 1kHZ | hearing_threshold_1khz | audiometry | Pure-tone hearing threshold at 1 kHz. | dB HL expected; unit not confirmed | clinical_metadata_no_values_exported | candidate_progression_biomarker | unit_unconfirmed |
| 2kHZ | hearing_threshold_2khz | audiometry | Pure-tone hearing threshold at 2 kHz. | dB HL expected; unit not confirmed | clinical_metadata_no_values_exported | candidate_progression_biomarker | unit_unconfirmed |
| 3kHZ | hearing_threshold_3khz | audiometry | Pure-tone hearing threshold at 3 kHz. | dB HL expected; unit not confirmed | clinical_metadata_no_values_exported | candidate_progression_biomarker | unit_unconfirmed |
| PTA | pta_aaohns_0_5_1_2_3khz | audiometry | Four-frequency mean of 0.5, 1, 2, and 3 kHz in the ear-level sheets. | dB HL expected; recompute from frequencies | clinical_metadata_no_values_exported | candidate_progression_biomarker | formula_audited_unit_unconfirmed |
| DHI-F | dhi_functional | vertigo_disability | Dizziness Handicap Inventory functional subscore. | score | clinical_metadata_no_values_exported | clinical_validator | instrument_version_unconfirmed |
| DHI-E | dhi_emotional | vertigo_disability | Dizziness Handicap Inventory emotional subscore. | score | clinical_metadata_no_values_exported | clinical_validator | instrument_version_unconfirmed |
| DHI-P | dhi_physical | vertigo_disability | Dizziness Handicap Inventory physical subscore. | score | clinical_metadata_no_values_exported | clinical_validator | instrument_version_unconfirmed |
| DHI-T | dhi_total | vertigo_disability | Dizziness Handicap Inventory total score. | score | clinical_metadata_no_values_exported | clinical_validator_not_irreversible_event | instrument_version_unconfirmed |
| THI | thi_total | tinnitus_handicap | Tinnitus Handicap Inventory source score. | score | clinical_metadata_no_values_exported | clinical_validator_not_irreversible_event | instrument_version_unconfirmed |
| 耳闷 | aural_fullness | symptom | Aural fullness source field. | binary/ordinal/text coding requires codebook | clinical_metadata_no_values_exported | clinical_validator_not_irreversible_event | unresolved_coding |
| VADL | vadl_total | activities_daily_living | Vestibular Activities of Daily Living source score. | score; version/range require codebook | clinical_metadata_no_values_exported | clinical_validator_not_irreversible_event | instrument_details_unconfirmed |
