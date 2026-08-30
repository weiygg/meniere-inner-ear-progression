# Segmentation V2 experiment summary

Primary ranking is reserved for patient-level five-fold OOF macro Dice on
`LS_SEG_200`. No formal E1-E8 OOF row exists yet, so no V2 winner is selected.

| Experiment | Architecture | Internal result | Status |
|---|---|---:|---|
| E0 | Three independent binary TinyViT-UNets | historical internal test 0.8136 | completed, not five-fold OOF |
| M1 | Joint 4-class PlainConvUNet | fold-0 five-epoch pilot 0.7964 | completed, not OOF |
| M2 | M1 plus bounded bias-field augmentation | fold-0 five-epoch pilot 0.7984 | completed, not OOF |
| E1 | Joint 4-class nnU-Net v2 | — | preprocessing/smoke completed; OOF not run |
| E2 | Residual Encoder nnU-Net | — | CUDA probe completed; OOF not run |
| E4 | High-depth-resolution joint nnU-Net plus MRI-robust augmentation | — | CUDA probe completed; OOF not run |
| E5 | E4 plus 0.30 boundary-aware loss | — | loss gradient smoke completed; OOF not run |
| E6 | E4 plus boundary-aware and foreground soft-clDice losses | — | loss gradient smoke completed; OOF not run |
| E7/E8 | Internal winner, five-fold probability ensemble, prespecified TTA | — | blocked by OOF training |

The local ignored `results/experiment_summary.csv` contains the full requested
schema and explicit blank values for experiments that have not run. The paired
improvement waterfall intentionally contains no deltas until matching OOF
ablations exist.
