# SEGMENTATION V2 baseline audit

Date: 2026-08-30

Latest audited development commit: `80c2c31e0c80cd3f6676f2a4c7b8ceb90965ca5a` on `codex/protocol-v2-seg-pebm`. `origin/main` is older and is not the source of the completed M1-M3 results.

## V1 architecture and input

- Architecture: three independent binary 3D TinyViT-UNet models (SSC, HSC, PSC), each with one input and one output channel.
- Trainable parameters per structure model: 1,722,977.
- Resampling: image interpolation order 1 in the legacy crop pipeline; masks use nearest-neighbour/order 0. Target spacing is 0.3472222 x 0.3472222 x 0.5 mm.
- Normalisation: nonzero 0.5th-99.5th percentile clipping followed by nonzero z-score normalisation.
- Crop: 128 x 128 x 48 voxels, centred by the frozen union localiser.

| Network point | X x Y x Z |
|---|---:|
| Input / stem | 128 x 128 x 48 |
| Encoder down1, stride 2 | 64 x 64 x 24 |
| Patch embedding, kernel/stride 4 | 16 x 16 x 6 |
| Transformer bottleneck | 16 x 16 x 6 (1,536 tokens) |
| Decoder output | 128 x 128 x 48 |

The z direction is reduced to six bottleneck tokens. This is a plausible thin-tubular-structure information bottleneck, but it is a hypothesis until a paired internal high-resolution ablation is run.

## Training definition

- Loss: 0.70 Dice + 0.20 Tversky + 0.10 focal. The Tversky denominator weights FP=0.65 and FN=0.35.
- Augmentation: translation up to four voxels, +/-10% intensity scale/shift, and Gaussian noise (SD 0.03). No scanner-resolution, blur, Rician-like noise, rotation or bias-field augmentation was present in V1.
- Split: patient-level 140/30/30 people (280/60/60 ears), seed 42. Both ears stay together. The internal test has been reviewed previously and is not an untouched test.
- Thresholds/postprocessing selected on validation only: SSC 0.15/top-1 component; HSC 0.20/all components; PSC 0.10/top-1 component.

## Verified V1 performance

| Cohort | SSC Dice | HSC Dice | PSC Dice | Macro Dice | Surface Dice 1 mm | ASSD, mm | HD95, mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lishui historical internal test | 0.794 | 0.837 | 0.810 | 0.8136 | not available in the frozen V1 table | 0.092 | 0.492 |
| Exposed external manual set | 0.655 | 0.712 | 0.639 | 0.6688 | 0.975 | 0.207 | 0.797 |

## External quantitative error audit

| Structure | Volume ratio mean (SD) | Median (IQR) | Precision | Recall | Interpretation |
|---|---:|---:|---:|---:|---|
| SSC | 1.165 (0.198) | 1.162 (1.057-1.267) | 0.616 | 0.711 | predominantly over-segmentation / false positives |
| HSC | 1.054 (0.167) | 1.030 (0.955-1.140) | 0.703 | 0.730 | mixed boundary-thickness and endpoint mismatch; no single FP/FN direction dominates |
| PSC | 1.178 (0.205) | 1.187 (1.061-1.311) | 0.599 | 0.697 | predominantly over-segmentation / false positives |

High 1-mm surface Dice with much lower volumetric Dice is compatible with one- to two-voxel thickness, boundary and endpoint differences in these small canals. It does not prove that thickness is the only cause. Motion, contrast, acquisition resolution and annotation convention remain candidate contributors.

## Reference-mask volume distribution

Ear-level summaries are descriptive. Shift tests use patient-mean volumes across both ears and Benjamini-Hochberg correction.

| Structure | Lishui median (IQR), mm3 | External median (IQR), mm3 | External/Lishui median ratio | BH q |
|---|---:|---:|---:|---:|
| SSC | 16.19 (14.35-18.21) | 12.72 (11.48-14.12) | 0.782 | 3.81e-13 |
| HSC | 22.09 (19.41-25.38) | 18.30 (15.52-20.41) | 0.825 | 8.09e-10 |
| PSC | 17.81 (15.94-20.09) | 14.65 (13.13-17.07) | 0.815 | 6.88e-10 |

## Five leading limitations to test

1. The isotropic 8x encoder reduction leaves only six z tokens at the transformer bottleneck.
2. Three independent binary models cannot explicitly exploit the fixed SSC/HSC/PSC spatial relationship.
3. The current Tversky term penalises FP more than FN; its effect must be tested against symmetric and FN-heavier variants on internal OOF predictions.
4. V1 augmentation is narrow for cross-scanner MRI and lacks resolution, blur and bias-field simulation.
5. The single historical 140/30/30 split gives less stable model-selection evidence than patient-level five-fold OOF evaluation.

## Executable experiment set on this workstation

- COMPLETED: V1 source/result audit; V1 internal and exposed-external quantitative error summary.
- COMPLETED previously: nnU-Net v2 multiclass fold-0 five-epoch pilots M1/M2/M3 on the locked internal validation split.
- EXECUTABLE: construct a 200-patient/400-ear five-fold split and Dataset502; validate nnU-Net planning and run smoke tests.
- COMPUTE-LIMITED: full five-fold training. One nnU-Net epoch previously required about 51 minutes on the 6-GB GTX 1660 Ti, so even a five-epoch x five-fold benchmark is approximately 21 GPU-hours before ablations.
- NOT RUN until internal winner is frozen: any further external-label evaluation. The existing 50 cases are already exposed and cannot serve as a new confirmatory cohort.

## Leakage audit

- Partition: PASS for the historical split; both ears remain with the patient. A new patient-level five-fold split is required for V2 OOF.
- Fit-on-training: PASS for augmentation and normalisation implementation; crop localisation uses a frozen deployment localiser.
- Tuning: PASS for recorded V1 thresholds; selected on Lishui validation only. External labels are prohibited for V2 selection.
- Input integrity: PASS; the model receives T2 crop only.
- Evaluation: PARTIAL; CIs and failure analysis exist, but formal V2 five-fold OOF and a new untouched external cohort are not yet available.
