# M1 five-epoch internal validation pilot

Date: 2026-08-28

M1 is an nnU-Net v2 2.6.2 `3d_fullres` multiclass model trained for five epochs
under the resource-constrained equal-budget pilot. The locked split contains 280
training ears and 60 validation ears from 30 people. Confidence intervals use
5,000 patient-clustered bootstrap samples, preserving both ears from each person.

| Metric | Dice | Patient-clustered 95% CI |
|---|---:|---:|
| SSC | 0.7809 | 0.7686-0.7923 |
| HSC | 0.8169 | 0.8049-0.8285 |
| PSC | 0.7912 | 0.7828-0.7998 |
| Macro | 0.7964 | 0.7882-0.8046 |

This is internal validation performance from a deliberately short implementation
pilot. It is not the 60-ear internal benchmark, not external validation, and not
evidence that external Dice has reached 0.78. The existing 50-person external set
was not loaded and remains unavailable for model, augmentation, loss, threshold,
checkpoint, or postprocessing selection.

The public machine-readable manifest contains aggregate estimates and hashes only.
Predictions, case-level metrics, source paths, masks, and model weights remain local.
