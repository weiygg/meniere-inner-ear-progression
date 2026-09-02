# Formal E1 fold-0 exposed-external benchmark

Date: 2026-09-02

## Execution decision

Formal five-fold training was paused during fold 1 at the user's request to obtain
a faster result. The completed fold-0 E1 checkpoint was fixed after 54 epochs and
before external inference. Its SHA-256 was
`689deb46e5628393591334793576516a6901ac39d98703376a3b1d86ae074c0e`.
The checkpoint, preprocessing, multiclass decoding and full-grid restoration were
not modified in response to external labels.

## Internal fold-0 result

The patient-level fold contained 40 people, 80 ears and 240 structure masks.

| Structure | Dice | Surface Dice at 0.5 mm | ASSD, mm | HD95, mm |
|---|---:|---:|---:|---:|
| SSC | 0.7958 | 0.9799 | 0.1185 | 0.4447 |
| HSC | 0.8360 | 0.9919 | 0.1041 | 0.3964 |
| PSC | 0.8141 | 0.9807 | 0.1109 | 0.4558 |
| Macro | **0.8153** | **0.9842** | **0.1112** | **0.4323** |

This is one internal fold, not five-fold out-of-fold performance.

## Previously exposed external benchmark

Confidence intervals used 5,000 patient-clustered bootstrap repetitions.

| Cohort | People / ears | SSC Dice | HSC Dice | PSC Dice | Macro Dice (95% CI) |
|---|---:|---:|---:|---:|---:|
| Center 2 stratum | 25 / 50 | 0.6802 | 0.6880 | 0.6214 | **0.6632 (0.6332-0.6939)** |
| Center 3 stratum | 25 / 50 | 0.6841 | 0.7475 | 0.6691 | **0.7003 (0.6870-0.7130)** |
| Pooled | 50 / 100 | 0.6822 | 0.7178 | 0.6453 | **0.6817 (0.6643-0.6987)** |

Pooled surface metrics were HD95 0.6434 mm, ASSD 0.2085 mm and Surface Dice at
1 mm 0.9876.

## Paired comparison with the prior M2 checkpoint

The comparison used the same 50 people, 100 ears and 300 masks. Positive values
favor E1.

| Scope | E1 minus M2 Macro Dice (95% patient-bootstrap CI) |
|---|---:|
| Center 2 | +0.0078 (+0.0028 to +0.0132) |
| Center 3 | +0.0165 (+0.0091 to +0.0252) |
| Pooled | **+0.0122 (+0.0074 to +0.0175)** |

Pooled structure-specific differences were SSC +0.0058 (-0.0005 to +0.0124),
HSC +0.0198 (+0.0128 to +0.0272), and PSC +0.0109 (+0.0046 to +0.0177).

## Interpretation boundary

The pooled external Macro Dice increased from 0.6696 for M2 to 0.6817 for E1,
but the prespecified target of 0.78 was not reached. Center 2 and Center 3 are two
previously exposed strata from the same institution, not two independent external
hospitals. This is an exploratory locked-benchmark comparison, not a new
confirmatory external validation. No further parameter selection may use these
external labels; a new untouched cohort is required for a confirmatory claim.

Only aggregate metrics and code are published. Images, masks, predictions,
patient-level metrics and model weights remain local.
