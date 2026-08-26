# ChatGPT review guide

Use the GitHub repository as a code-and-aggregate-results review source. It deliberately
does not contain patient-level data, images, masks, model weights, or absolute local
paths.

After authorizing the GitHub connector in ChatGPT, select
`weiygg/meniere-inner-ear-progression` and ask:

> Read `docs/EXTERNAL_DICE_REAUDIT_20260826.md` and the five linked segmentation
> scripts. Propose a prespecified improvement experiment that uses only the internal
> training/validation data for model and hyperparameter selection. Do not tune on the
> existing 50-case external labels or call an adapted result independent validation.
> Prioritize nnU-Net versus the existing TinyViT-UNet, scanner-robust augmentation,
> topology/centerline-aware losses, and a blinded inter-annotator study. Return a
> concrete implementation plan, configuration diff, ablation table shell, compute
> estimate for a 6-GB GPU, and a locked acceptance rule for a new untouched external
> cohort with target macro Dice at least 0.78.

For a newly updated repository, GitHub connector indexing may take several minutes.
Public web access and GitHub connector authorization are separate; use the connector
when code-aware cross-file review is required.
