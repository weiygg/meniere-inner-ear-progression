# External semicircular-canal Dice re-audit

Date: 2026-08-26

## Review question

How should the SSC/HSC/PSC segmentation pipeline be improved toward a prespecified
external macro Dice target of at least 0.78 without tuning on the existing external
test labels?

## Verified result

The frozen model was evaluated on 50 external patients, 100 ears, and 300 manual
SSC/HSC/PSC masks. Patient-clustered bootstrap results were:

| Endpoint | Estimate | 95% CI |
|---|---:|---:|
| Macro Dice | 0.6688 | 0.6501–0.6871 |
| SSC Dice | 0.6553 | 0.6301–0.6788 |
| HSC Dice | 0.7115 | 0.6940–0.7287 |
| PSC Dice | 0.6395 | 0.6137–0.6638 |
| Macro surface Dice at 1 mm | 0.9753 | 0.9650–0.9846 |
| Macro ASSD, mm | 0.2072 | 0.1700–0.2519 |
| Macro HD95, mm | 0.7974 | 0.6606–0.9522 |

The internal patient-level test macro Dice was 0.8136 (95% CI 0.8053–0.8214).
Values near 0.81–0.82 elsewhere in the repository are internal test or union-localizer
results, not external manual-reference Dice.

## Re-audit findings

- Independent recomputation matched the formal 300-mask table with maximum absolute
  error `5.55e-17`.
- The frozen formal version was the best audited historical external version:
  archived v2 macro Dice was 0.6404 and legacy v1 was 0.4013.
- The all-T2 model is not comparable because it predicts cochlea, total vestibule,
  and vestibular structures rather than SSC/HSC/PSC.
- The identity SSC/HSC/PSC mapping was optimal for all 100 ears.
- Swapping left and right improved none of the 50 patients.
- A diagnostic-only post hoc search over integer shifts of ±2 voxels increased mean
  Dice only to 0.7168. This is not a reportable independent-validation result.
- Manual and inference-source T2 volumes were on the same grid and had image
  correlations of approximately 1.0. A systematic case-pairing or rigid-registration
  error was not found.

The combination of high 1-mm surface Dice and substantially lower volumetric Dice
suggests that one- to two-voxel boundary, thickness, or segment-length differences in
small tubular structures contribute strongly. This is a diagnosis, not proof of a
specific causal mechanism.

## Validation boundary

The external labels must not be used to select thresholds, morphology rules,
registration shifts, augmentation policies, architecture, or checkpoints while the
same 50 cases retain an independent external-validation claim. Patient-level tables,
images, masks, model weights, and per-mask results are intentionally excluded from
this repository.

## Requested technical review

Please inspect the following code paths and propose a small, prespecified comparison
that can be selected exclusively on the Lishui training/validation data:

- `src/39_train_structure_specific_vit_ensemble.py`
- `src/41_calibrate_ensemble_postprocessing.py`
- `src/43_infer_z2_structure_ensemble.py`
- `src/53_evaluate_external_manual_masks.py`
- `scripts/54_reaudit_external_dice_versions.py`

Priority questions:

1. Would nnU-Net be a stronger primary benchmark than the present structure-specific
   TinyViT-UNet under the available 3D crop size and anisotropic spacing?
2. Which scanner-robust augmentations can be fixed a priori and selected only by the
   internal validation set?
3. Should topology- or centerline-aware losses be added for these thin tubular masks?
4. What blinded inter-annotator experiment is sufficient to distinguish model domain
   shift from annotation-protocol shift?
5. What minimum new untouched external cohort and acceptance rule should be locked
   before testing the target macro Dice of at least 0.78?

Aggregate machine-readable values are provided in
`docs/external_dice_reaudit_summary_20260826.json`.
