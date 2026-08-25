# Hearing-based affected-ear rule sensitivity audit

Status: **exploratory sensitivity analysis; no affected-ear rule is frozen here, and no P-EBM was run.**

## Scope and safeguards

- Unit: one baseline patient with a complete left-right audiometric pair.
- Audiometry: thresholds at 0.5, 1, 2, and 3 kHz; the four-frequency PTA is recomputed in code.
- Abnormality is defined using strict `>` comparisons at 20, 25, 30, 35, or 40 dB HL.
- The grid contains 30 ear-level rules and 5 bilateral-resolution modes (150 combinations per cohort).
- AAO-HNS-stage-cell presence is evaluated only as an obsolete comparator, never as the affected-ear definition.
- The report and JSON contain aggregate counts only; no patient key, name, record number, or row-level assignment is exported.

The [WHO World Report on Hearing](https://www.who.int/publications/i/item/9789240020481) uses a 0.5/1/2/4-kHz average for population hearing-loss grading, so its 20/35-dB boundaries are not directly reconstructable or adopted here because this workbook has 3 kHz rather than 4 kHz. The AAO-HNS 1995 guideline is documented in [PubMed](https://pubmed.ncbi.nlm.nih.gov/7675476/); a later clinical paper describes its 0.5/1/2/3-kHz arithmetic mean ([PubMed](https://pubmed.ncbi.nlm.nih.gov/12925337/)). These sources motivate sensitivity anchors only; neither source resolves this study's bilateral index-ear definition.

## Cohort flow

| Cohort | Baseline patients | Incomplete/duplicate pair | Incomplete four-frequency pair | Analysed paired patients |
|---|---:|---:|---:|---:|
| LS_baseline_all | 79 | 0 | 23 | 56 |
| Z2_confirmed_MD_baseline | 96 | 0 | 6 | 90 |

## Prespecified summary subset

The complete 150-combination results per cohort are in the companion aggregate JSON. This table shows all PTA thresholds plus two interpretable cross-family comparators.

### LS_baseline_all

| Rule | Bilateral handling | One abnormal | Both abnormal | Neither abnormal | Resolved index ear | Resolved % (95% Wilson CI) |
|---|---|---:|---:|---:|---:|---:|
| pta_05123_gt_20 | strict | 12 | 43 | 1 | 12 | 21.4% (12.7%-33.8%) |
| pta_05123_gt_20 | pta_gap_10 | 12 | 43 | 1 | 43 | 76.8% (64.2%-85.9%) |
| pta_05123_gt_20 | pta_gap_15 | 12 | 43 | 1 | 40 | 71.4% (58.5%-81.6%) |
| pta_05123_gt_20 | pta_gap_20 | 12 | 43 | 1 | 34 | 60.7% (47.6%-72.4%) |
| pta_05123_gt_20 | worse_pta_no_minimum | 12 | 43 | 1 | 55 | 98.2% (90.6%-99.7%) |
| pta_05123_gt_25 | strict | 21 | 33 | 2 | 21 | 37.5% (26.0%-50.6%) |
| pta_05123_gt_25 | pta_gap_10 | 21 | 33 | 2 | 44 | 78.6% (66.2%-87.3%) |
| pta_05123_gt_25 | pta_gap_15 | 21 | 33 | 2 | 41 | 73.2% (60.4%-83.0%) |
| pta_05123_gt_25 | pta_gap_20 | 21 | 33 | 2 | 37 | 66.1% (53.0%-77.1%) |
| pta_05123_gt_25 | worse_pta_no_minimum | 21 | 33 | 2 | 54 | 96.4% (87.9%-99.0%) |
| pta_05123_gt_30 | strict | 25 | 25 | 6 | 25 | 44.6% (32.4%-57.6%) |
| pta_05123_gt_30 | pta_gap_10 | 25 | 25 | 6 | 42 | 75.0% (62.3%-84.5%) |
| pta_05123_gt_30 | pta_gap_15 | 25 | 25 | 6 | 39 | 69.6% (56.7%-80.1%) |
| pta_05123_gt_30 | pta_gap_20 | 25 | 25 | 6 | 35 | 62.5% (49.4%-74.0%) |
| pta_05123_gt_30 | worse_pta_no_minimum | 25 | 25 | 6 | 50 | 89.3% (78.5%-95.0%) |
| pta_05123_gt_35 | strict | 30 | 17 | 9 | 30 | 53.6% (40.7%-66.0%) |
| pta_05123_gt_35 | pta_gap_10 | 30 | 17 | 9 | 41 | 73.2% (60.4%-83.0%) |
| pta_05123_gt_35 | pta_gap_15 | 30 | 17 | 9 | 39 | 69.6% (56.7%-80.1%) |
| pta_05123_gt_35 | pta_gap_20 | 30 | 17 | 9 | 37 | 66.1% (53.0%-77.1%) |
| pta_05123_gt_35 | worse_pta_no_minimum | 30 | 17 | 9 | 47 | 83.9% (72.2%-91.3%) |
| pta_05123_gt_40 | strict | 30 | 13 | 13 | 30 | 53.6% (40.7%-66.0%) |
| pta_05123_gt_40 | pta_gap_10 | 30 | 13 | 13 | 39 | 69.6% (56.7%-80.1%) |
| pta_05123_gt_40 | pta_gap_15 | 30 | 13 | 13 | 37 | 66.1% (53.0%-77.1%) |
| pta_05123_gt_40 | pta_gap_20 | 30 | 13 | 13 | 35 | 62.5% (49.4%-74.0%) |
| pta_05123_gt_40 | worse_pta_no_minimum | 30 | 13 | 13 | 43 | 76.8% (64.2%-85.9%) |
| freq_gt_25_n2 | strict | 23 | 32 | 1 | 23 | 41.1% (29.2%-54.1%) |
| freq_gt_25_n2 | pta_gap_10 | 23 | 32 | 1 | 46 | 82.1% (70.2%-90.0%) |
| freq_gt_25_n2 | pta_gap_15 | 23 | 32 | 1 | 43 | 76.8% (64.2%-85.9%) |
| freq_gt_25_n2 | pta_gap_20 | 23 | 32 | 1 | 39 | 69.6% (56.7%-80.1%) |
| freq_gt_25_n2 | worse_pta_no_minimum | 23 | 32 | 1 | 55 | 98.2% (90.6%-99.7%) |
| lowfreq_mean_gt_25 | strict | 24 | 31 | 1 | 24 | 42.9% (30.8%-55.9%) |
| lowfreq_mean_gt_25 | pta_gap_10 | 24 | 31 | 1 | 47 | 83.9% (72.2%-91.3%) |
| lowfreq_mean_gt_25 | pta_gap_15 | 24 | 31 | 1 | 44 | 78.6% (66.2%-87.3%) |
| lowfreq_mean_gt_25 | pta_gap_20 | 24 | 31 | 1 | 39 | 69.6% (56.7%-80.1%) |
| lowfreq_mean_gt_25 | worse_pta_no_minimum | 24 | 31 | 1 | 55 | 98.2% (90.6%-99.7%) |

Rule-family envelopes across all thresholds/count variants:

| Family | Bilateral handling | Minimum resolved | Maximum resolved |
|---|---|---:|---:|
| frequency_count | pta_gap_10 | 34 | 46 |
| frequency_count | pta_gap_15 | 34 | 43 |
| frequency_count | pta_gap_20 | 32 | 40 |
| frequency_count | strict | 6 | 34 |
| frequency_count | worse_pta_no_minimum | 36 | 55 |
| low_frequency_mean | pta_gap_10 | 43 | 47 |
| low_frequency_mean | pta_gap_15 | 41 | 44 |
| low_frequency_mean | pta_gap_20 | 34 | 40 |
| low_frequency_mean | strict | 14 | 32 |
| low_frequency_mean | worse_pta_no_minimum | 46 | 55 |
| pta_05123 | pta_gap_10 | 39 | 44 |
| pta_05123 | pta_gap_15 | 37 | 41 |
| pta_05123 | pta_gap_20 | 34 | 37 |
| pta_05123 | strict | 12 | 30 |
| pta_05123 | worse_pta_no_minimum | 43 | 55 |

Cross-rule consensus among all 30 ear-level rules:

| Bilateral handling | All resolve + agree | All resolve + disagree | Partial + agree | Partial + disagree | None resolve |
|---|---:|---:|---:|---:|---:|
| strict | 4 | 0 | 43 | 2 | 7 |
| pta_gap_10 | 33 | 0 | 18 | 2 | 3 |
| pta_gap_15 | 31 | 0 | 20 | 2 | 3 |
| pta_gap_20 | 28 | 0 | 22 | 2 | 4 |
| worse_pta_no_minimum | 36 | 0 | 17 | 2 | 1 |
### Z2_confirmed_MD_baseline

| Rule | Bilateral handling | One abnormal | Both abnormal | Neither abnormal | Resolved index ear | Resolved % (95% Wilson CI) |
|---|---|---:|---:|---:|---:|---:|
| pta_05123_gt_20 | strict | 32 | 53 | 5 | 32 | 35.6% (26.4%-45.8%) |
| pta_05123_gt_20 | pta_gap_10 | 32 | 53 | 5 | 73 | 81.1% (71.8%-87.9%) |
| pta_05123_gt_20 | pta_gap_15 | 32 | 53 | 5 | 68 | 75.6% (65.8%-83.3%) |
| pta_05123_gt_20 | pta_gap_20 | 32 | 53 | 5 | 60 | 66.7% (56.4%-75.5%) |
| pta_05123_gt_20 | worse_pta_no_minimum | 32 | 53 | 5 | 85 | 94.4% (87.6%-97.6%) |
| pta_05123_gt_25 | strict | 43 | 39 | 8 | 43 | 47.8% (37.8%-58.0%) |
| pta_05123_gt_25 | pta_gap_10 | 43 | 39 | 8 | 71 | 78.9% (69.4%-86.0%) |
| pta_05123_gt_25 | pta_gap_15 | 43 | 39 | 8 | 66 | 73.3% (63.4%-81.4%) |
| pta_05123_gt_25 | pta_gap_20 | 43 | 39 | 8 | 60 | 66.7% (56.4%-75.5%) |
| pta_05123_gt_25 | worse_pta_no_minimum | 43 | 39 | 8 | 82 | 91.1% (83.4%-95.4%) |
| pta_05123_gt_30 | strict | 50 | 27 | 13 | 50 | 55.6% (45.3%-65.4%) |
| pta_05123_gt_30 | pta_gap_10 | 50 | 27 | 13 | 69 | 76.7% (66.9%-84.2%) |
| pta_05123_gt_30 | pta_gap_15 | 50 | 27 | 13 | 65 | 72.2% (62.2%-80.4%) |
| pta_05123_gt_30 | pta_gap_20 | 50 | 27 | 13 | 59 | 65.6% (55.3%-74.6%) |
| pta_05123_gt_30 | worse_pta_no_minimum | 50 | 27 | 13 | 77 | 85.6% (76.8%-91.4%) |
| pta_05123_gt_35 | strict | 52 | 20 | 18 | 52 | 57.8% (47.5%-67.5%) |
| pta_05123_gt_35 | pta_gap_10 | 52 | 20 | 18 | 67 | 74.4% (64.6%-82.3%) |
| pta_05123_gt_35 | pta_gap_15 | 52 | 20 | 18 | 64 | 71.1% (61.0%-79.5%) |
| pta_05123_gt_35 | pta_gap_20 | 52 | 20 | 18 | 59 | 65.6% (55.3%-74.6%) |
| pta_05123_gt_35 | worse_pta_no_minimum | 52 | 20 | 18 | 72 | 80.0% (70.6%-87.0%) |
| pta_05123_gt_40 | strict | 49 | 17 | 24 | 49 | 54.4% (44.2%-64.3%) |
| pta_05123_gt_40 | pta_gap_10 | 49 | 17 | 24 | 61 | 67.8% (57.6%-76.5%) |
| pta_05123_gt_40 | pta_gap_15 | 49 | 17 | 24 | 59 | 65.6% (55.3%-74.6%) |
| pta_05123_gt_40 | pta_gap_20 | 49 | 17 | 24 | 55 | 61.1% (50.8%-70.5%) |
| pta_05123_gt_40 | worse_pta_no_minimum | 49 | 17 | 24 | 66 | 73.3% (63.4%-81.4%) |
| freq_gt_25_n2 | strict | 47 | 35 | 8 | 47 | 52.2% (42.0%-62.2%) |
| freq_gt_25_n2 | pta_gap_10 | 47 | 35 | 8 | 73 | 81.1% (71.8%-87.9%) |
| freq_gt_25_n2 | pta_gap_15 | 47 | 35 | 8 | 69 | 76.7% (66.9%-84.2%) |
| freq_gt_25_n2 | pta_gap_20 | 47 | 35 | 8 | 62 | 68.9% (58.7%-77.5%) |
| freq_gt_25_n2 | worse_pta_no_minimum | 47 | 35 | 8 | 82 | 91.1% (83.4%-95.4%) |
| lowfreq_mean_gt_25 | strict | 56 | 22 | 12 | 56 | 62.2% (51.9%-71.5%) |
| lowfreq_mean_gt_25 | pta_gap_10 | 56 | 22 | 12 | 72 | 80.0% (70.6%-87.0%) |
| lowfreq_mean_gt_25 | pta_gap_15 | 56 | 22 | 12 | 69 | 76.7% (66.9%-84.2%) |
| lowfreq_mean_gt_25 | pta_gap_20 | 56 | 22 | 12 | 64 | 71.1% (61.0%-79.5%) |
| lowfreq_mean_gt_25 | worse_pta_no_minimum | 56 | 22 | 12 | 78 | 86.7% (78.1%-92.2%) |

Rule-family envelopes across all thresholds/count variants:

| Family | Bilateral handling | Minimum resolved | Maximum resolved |
|---|---|---:|---:|
| frequency_count | pta_gap_10 | 41 | 74 |
| frequency_count | pta_gap_15 | 40 | 69 |
| frequency_count | pta_gap_20 | 39 | 63 |
| frequency_count | strict | 26 | 54 |
| frequency_count | worse_pta_no_minimum | 44 | 86 |
| low_frequency_mean | pta_gap_10 | 58 | 74 |
| low_frequency_mean | pta_gap_15 | 55 | 69 |
| low_frequency_mean | pta_gap_20 | 53 | 64 |
| low_frequency_mean | strict | 47 | 56 |
| low_frequency_mean | worse_pta_no_minimum | 62 | 83 |
| pta_05123 | pta_gap_10 | 61 | 73 |
| pta_05123 | pta_gap_15 | 59 | 68 |
| pta_05123 | pta_gap_20 | 55 | 60 |
| pta_05123 | strict | 32 | 52 |
| pta_05123 | worse_pta_no_minimum | 66 | 85 |

Cross-rule consensus among all 30 ear-level rules:

| Bilateral handling | All resolve + agree | All resolve + disagree | Partial + agree | Partial + disagree | None resolve |
|---|---:|---:|---:|---:|---:|
| strict | 9 | 0 | 66 | 3 | 12 |
| pta_gap_10 | 41 | 0 | 38 | 3 | 8 |
| pta_gap_15 | 38 | 0 | 40 | 3 | 9 |
| pta_gap_20 | 33 | 0 | 44 | 3 | 10 |
| worse_pta_no_minimum | 44 | 0 | 39 | 3 | 4 |

## Interpretation boundary

A high resolved fraction or agreement with the legacy stage-cell proxy is not evidence that a rule is clinically correct. The final definition still requires the study team's hearing-feature specification, examination-time rule, and documented handling of bilateral disease. In particular, `worse_pta_no_minimum` is an exploratory stress test and is not eligible for automatic adoption.

Pairwise agreement and Cohen kappa for all 30 rules are provided for `strict` and `pta_gap_15` in the JSON. Agreement is calculated only among patients resolved by both rules and therefore must be read together with the common resolved denominator.
