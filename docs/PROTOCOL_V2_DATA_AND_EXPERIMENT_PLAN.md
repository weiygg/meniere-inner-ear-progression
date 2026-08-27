# Protocol V2 data and experiment plan

## Two independent research tracks

```text
LS_SEG_200 (200 people / 400 ears)
  -> patient-level segmentation development
  -> M0-M3 internal-only model selection
  -> frozen model
  -> locked external benchmark

LS_CLIN_79 (79 people / 158 ear rows) + Z2_CLIN
  -> namespaced clinical QC
  -> frozen segmentation inference
  -> geometry and objective biomarkers
  -> common-core P-EBM eligibility
  -> pseudo-temporal ordering, validators, and endotypes
```

`LS_SEG_200` and `LS_CLIN_79` independently restart numeric IDs at 1. The study
team confirmed patient overlap is 0. Persistent identifiers must therefore use
dataset namespaces; a numeric match is never identity evidence.

## Segmentation experiment sequence

| Stage | Model/action | Selection data | External labels allowed? |
|---|---|---|---|
| A0 | Audit frozen M0 and split provenance | metadata only | no selection |
| A1-A2 | M1 nnU-Net v2 multiclass and internal evaluation | LS_SEG_200 | no |
| A3-A4 | M2 fixed scanner-robust augmentation | LS_SEG_200 | no |
| A5-A6 | M3 M2 + fixed soft-clDice component | LS_SEG_200 | no |
| A7-A8 | Freeze internal winner and model card | LS_SEG_200 | no |
| A9 | Evaluate current 50-case locked benchmark | EXT_MANUAL_50 | evaluation only |
| A10 | Manual-vs-AI geometry reliability | EXT_MANUAL_50 | reliability only |
| A11 | One-shot new confirmatory evaluation | future untouched cohort | labels opened only after lock |

The current 50-person benchmark has already been exposed. It cannot select any model
decision and cannot be described as a new untouched confirmatory validation after
redevelopment.

## M0-M3 prespecification

- M0: frozen structure-specific binary 3D TinyViT-UNet baseline.
- M1: nnU-Net v2 `3d_fullres`, one multiclass model (background/SSC/HSC/PSC).
- M2: M1 plus a fixed first-layer scanner-robust augmentation set.
- M3: M2 plus Dice/CE and a fixed soft-clDice weight of 0.1.

The 400-ear source uses three binary masks. An aggregate audit found 2,565 shared
boundary voxels in 195 ears (2,551 SSC-PSC, 14 SSC-HSC, no triple overlap). M1 uses
the frozen nearest-exclusive-core conversion in physical space; source masks remain
read-only and the conversion fails if no explicit overlap policy is supplied.

The locked nnU-Net dataset contains 280 training ears and 60 validation ears. The
60-ear previously viewed internal benchmark is stored separately and is excluded
from fingerprint extraction, planning, augmentation/loss choice, and threshold
selection.

Internal selection is ordered: architecture, augmentation, topology loss, then
threshold/postprocessing. The primary metric is macro Dice across SSC/HSC/PSC;
surface, volume, and topology metrics are co-reported.

## Clinical gates

Final P-EBM is prohibited until the explicit Z2 crosswalk, signed codebook, dB HL and
measurement-window confirmation, and affected/index-ear rules are available. DHI,
THI, VADL, ear fullness, symptom burden, AAO-HNS stage, and static geometry are not
default irreversible primary events. P-EBM estimates a pseudo-temporal ordering of
biomarker abnormalities, not longitudinal transitions or causality.

## Reproducibility and privacy

Every phase writes a Git-ignored run directory with config hashes, Git commit,
package versions, seeds, output hashes, warnings, and blockers. Public Git contains
code, schemas, aggregate counts, hashes, and methodological documents only.
