# Cohort definition - audit checkpoint draft

Status: **not yet adjudicated**. This file records verified source structure and the
intended unit; it is not a frozen cohort specification.

## Confirmed affected-ear principle

Affected-ear status is determined from abnormal clinical hearing features. It must
not be inferred from whether the AAO-HNS stage cell is populated. The exact hearing
features, abnormal thresholds, and examination-time rule still require a frozen
definition before cohort generation.

## Intended primary unit

The primary P-EBM unit will be one index diseased ear per patient. The intended
decision sequence is:

1. classify each ear using the prespecified clinical hearing-feature rule;
2. if exactly one ear is abnormal, use that clinically affected ear;
3. if both ears are abnormal, use the documented first affected or index Ménière ear;
4. if the bilateral index ear is undocumented, stop or apply a separately approved
   sensitivity rule rather than selecting an ear post hoc.

The numerically worse-hearing ear will not be selected merely to simplify the
dataset. Both ears will remain in the canonical database. Any all-ear sensitivity
analysis will retain patient clustering for bootstrap and regression.

## Hearing-rule sensitivity audit (not a frozen definition)

`reports/01_hearing_ear_rule_sensitivity.md` compares 30 hearing-abnormality rules
under five bilateral-resolution modes. It uses complete left-right baseline pairs,
reports only aggregate results, and does not run P-EBM. Its purpose is to quantify
how much the cohort changes under plausible definitions; rule yield, cross-rule
agreement, or agreement with the obsolete stage-cell proxy must not be used alone
to declare clinical correctness.

## Verified source counts

| Step | People / visits | Ear rows | Status |
|---|---:|---:|---|
| Lishui ear-level sheet | 79 baseline people | 158 | source structure verified |
| Z2 ear-level sheet | 103 visits | 206 | prior audit: 100 baseline people + 3 follow-ups |
| Combined source-prefixed baseline | 179 people | 358 | data-audit denominator only |
| First-audit unique index-ear proxy | 150 people | 150 | obsolete proxy count; must be recomputed using the hearing-feature rule |
| First-audit unresolved index ear | 29 people | 58 potential ear rows | historical proxy count, not the expected final unresolved count |
| Later Z2 confirmed-MD proposal | 96 baseline people | patient-aggregated | provisional local specification; not yet adopted |

## Frozen imaging-center split

| Role | Center/group | Protected source groups | Prior imaging denominator | Use |
|---|---|---|---:|---|
| Primary development | Lishui | `丽水-xjj内耳分割4` | 200 people / 400 ears | patient-level training and internal validation only |
| External validation 1 | Z2 group 1 | `浙二1-1`, `浙二1-2` | 51 imaging studies | frozen external test only |
| External validation 2 | Z2 group 2 | `浙二2-1`, `浙二2-2`, `浙二2例新` | 43 imaging studies | frozen external test only |

External validation 1 and 2 are separate prespecified strata from Zhejiang Second
Hospital, not two proven independent hospitals. Neither group may be used for
training, model selection, feature selection, thresholding, harmonization fitting,
or postprocessing decisions. The split is machine-readable in
`configs/center_split.yaml`.

## Required keys

- `patient_id`: stable, de-identified, source-namespaced person key;
- `ear_id`: `patient_id + ear_side`;
- `visit_id`: `patient_id + index_visit_date_or_sequence`;
- `measurement_id`: `visit_id + ear_side + test_type + replicate`.

No master patient/ear/visit/measurement CSV has been generated at this checkpoint.
Generation is blocked by `docs/OPEN_ISSUES.md` O001-O010.
