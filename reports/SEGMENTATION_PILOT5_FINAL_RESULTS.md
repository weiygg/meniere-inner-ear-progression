# Protocol V2 semicircular-canal segmentation results

Date: 2026-08-29

## Completion status

The M1/M2/M3 five-epoch equal-budget pilot is complete. All three models used the
same patient-level split, with 140 people/280 ears for training and 30 people/60
ears for internal validation. Training-set Dice is intentionally not used as a
performance estimate. All model selection occurred on the locked internal
validation set; external labels were not loaded during training, checkpointing,
or selection.

## Internal model selection

Confidence intervals use 5,000 patient-clustered bootstrap repetitions.

| Model | SSC Dice | HSC Dice | PSC Dice | Macro Dice (95% CI) | Selection |
|---|---:|---:|---:|---:|---|
| M1 nnU-Net v2 multiclass | 0.7809 | 0.8169 | 0.7912 | 0.7964 (0.7882-0.8046) | not selected |
| M2 + bounded MRI bias-field augmentation | 0.7849 | 0.8167 | 0.7937 | **0.7984 (0.7909-0.8058)** | selected among M1-M3 |
| M3 + 0.1 soft-clDice | 0.7801 | 0.8193 | 0.7877 | 0.7957 (0.7873-0.8038) | not selected |

M2 exceeded M1 by 0.0021 and M3 by 0.0028 in internal macro Dice. These small
point-estimate differences do not establish statistical superiority; M2 is the
prespecified point-estimate winner within this short implementation pilot.

## Frozen evaluation on the two exposed external strata

After internal selection was complete, the M2 final checkpoint was run once on
the two previously exposed 25-patient manual-reference subsets. The frozen union
localizer centers, 0.3472222 x 0.3472222 x 0.5-mm spacing, 128 x 128 x 48 crops,
nnU-Net checkpoint, and multiclass decoding were applied without external-label
parameter selection. All 100 ears produced non-empty SSC/HSC/PSC predictions.

| Cohort | People / ears | SSC Dice | HSC Dice | PSC Dice | Macro Dice (95% CI) |
|---|---:|---:|---:|---:|---:|
| Center 2 stratum | 25 / 50 | 0.6727 | 0.6763 | 0.6173 | **0.6554 (0.6256-0.6857)** |
| Center 3 stratum | 25 / 50 | 0.6801 | 0.7197 | 0.6514 | **0.6837 (0.6696-0.6973)** |
| Pooled exposed external | 50 / 100 | 0.6764 | 0.6980 | 0.6343 | **0.6696 (0.6527-0.6867)** |

Additional macro-average surface results were:

| Cohort | HD95, mm | ASSD, mm | Surface Dice at 1 mm |
|---|---:|---:|---:|
| Center 2 stratum | 0.7499 | 0.2477 | 0.9815 |
| Center 3 stratum | 0.8005 | 0.2287 | 0.9827 |
| Pooled exposed external | 0.7752 | 0.2382 | 0.9821 |

The external macro Dice target of 0.78 was not reached. The internally selected
M2 result was nearly unchanged from the historical frozen M0 pooled external
macro Dice (0.6696 vs 0.6688). Center 2 changed from 0.6505 to 0.6554 and Center 3
from 0.6871 to 0.6837; these are descriptive comparisons on the same exposed
cases, not independent model-comparison estimates.

## Interpretation boundary

Center 2 and Center 3 are predefined strata from the same external institution,
not two independent hospitals. Their labels had already been reviewed in earlier
audits, so this is a locked exposed-benchmark evaluation, not a new confirmatory
external validation. The result supports persistent external generalization loss
and does not support a claim that Dice exceeded 0.78. A new untouched external
cohort remains necessary for a confirmatory claim.

Only aggregate metrics and hashes are public. Images, masks, patient identifiers,
case-level metrics, local paths, predictions, and model weights remain local.
