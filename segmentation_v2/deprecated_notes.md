# Legacy implementation notes

No legacy file was deleted or overwritten. The V1 three-model TinyViT pipeline and
all historical checkpoints/results remain the reproducible historical baseline.

Known limitations are documented, not retroactively patched:

- the historical split is a 140/30/30 patient split rather than the new fixed
  patient-level five-fold design;
- each structure was trained independently, so shared anatomical relationships
  were not explicitly modelled;
- the deepest tensor retains only six voxels/tokens along depth;
- scanner-robust augmentation was limited compared with nnU-Net defaults plus the
  new bounded bias-field transform;
- historical external results are already exposed and cannot select V2 settings.

These limitations do not invalidate the archived V1 numbers; they define E0 and
the hypotheses tested by the isolated `segmentation_v2` implementation.
