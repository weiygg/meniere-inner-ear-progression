# Semicircular-canal segmentation V2

This directory isolates segmentation optimisation from the legacy V1 pipeline and
from the clinical P-EBM workflow. V1 scripts, checkpoints and results are preserved.

## Scientific boundary

- Development/model selection: `LS_SEG_200` only, using a fixed patient-level
  five-fold split. Both ears of a patient always remain in the same fold.
- Existing Zhejiang Second Hospital manual references: already exposed, from two
  prespecified batches at the same institution. They may be retained as a locked
  historical benchmark but cannot select architecture, loss, augmentation,
  checkpoint, TTA or postprocessing.
- Confirmatory external testing requires a new untouched cohort after model freeze.
- Patient-level split files, predictions, masks, images and weights remain local and
  Git-ignored. Only aggregate summaries may be published.

## Current status

- `SEGMENTATION_V2_BASELINE_AUDIT.md`: COMPLETED.
- Fixed 200-patient/400-ear five-fold split: COMPLETED locally; aggregate hash is in
  `data/manifests/segmentation_v2_cv_split_summary.json`.
- Dataset502 construction, integrity validation and nnU-Net preprocessing: COMPLETED
  locally for 200 patients/400 ears. The default 3D plan uses a 48 x 128 x 128
  (D/H/W) patch and anisotropic late-stage strides.
- CUDA smoke test: PASSED for the multiclass nnU-Net E1/M1-M3 trainer variants.
- Residual Encoder nnU-Net E2 planning: COMPLETED; a batch-1 mixed-precision GPU
  forward/backward probe is required before it is called trainable on 6 GB VRAM.
- Formal E1 five-fold OOF training: **RUNNING LOCALLY**. Fold 0 completed on
  2026-09-02 (40 patients/80 ears; internal Macro Dice 0.8153), and fold 1 is
  running. The workflow uses
  a fixed 54-epoch compute cap with the native nnU-Net 1000-epoch learning-rate
  schedule horizon, serial folds 0-4, mixed precision and a
  one-epoch checkpoint interval. Runtime state, logs, PIDs, predictions and weights
  are local/Git-ignored.
- `MODEL_FREEZE.md`: NOT FROZEN. External evaluation is blocked.

## Reproducible commands

All paths below are examples. Use local protected paths through command-line
arguments; do not add them to YAML or Git.

```powershell
python segmentation_v2\scripts\audit_v1.py --manifest <local-v1-manifest.csv> --internal-metrics <local-internal.csv> --internal-summary <local-summary.json> --external-metrics <local-external.csv> --output-dir segmentation_v2\results\baseline_audit --report segmentation_v2\SEGMENTATION_V2_BASELINE_AUDIT.md --public-json data\manifests\segmentation_v2_baseline_audit_20260830.json

python segmentation_v2\scripts\make_cv_split.py --manifest <local-v1-manifest.csv> --output segmentation_v2\splits\cv_split.csv --summary data\manifests\segmentation_v2_cv_split_summary.json --nnunet-json segmentation_v2\splits\splits_final.json

python segmentation_v2\scripts\prepare_dataset.py --manifest <local-v1-manifest.csv> --cv-split segmentation_v2\splits\cv_split.csv --nnunet-raw <ascii-raw-root>

python segmentation_v2\scripts\plan_and_preprocess.py --nnunet-raw <ascii-raw-root> --nnunet-preprocessed <ascii-preprocessed-root> --nnunet-results <ascii-results-root> --run-dir segmentation_v2\results\planning

python segmentation_v2\scripts\train_cv.py --config segmentation_v2\configs\nnunet.yaml --fold 0 --num-epochs 5 --dry-run

nnUNetv2_plan_experiment -d 502 -pl nnUNetPlannerResEncM

python segmentation_v2\scripts\probe_residual_gpu.py --nnunet-raw <ascii-raw-root> --nnunet-preprocessed <ascii-preprocessed-root> --nnunet-results <ascii-results-root> --output segmentation_v2\results\residual_gpu_probe.json
```

## Detached local formal training

The worker automatically skips completed folds, resumes `checkpoint_latest.pth`,
runs folds sequentially, evaluates each fold, and produces patient-clustered OOF
metrics after fold 4. It never loads external labels.

```powershell
.\segmentation_v2\check_training_status.ps1
.\segmentation_v2\stop_training.ps1
.\segmentation_v2\resume_training.ps1
```

`stop_training.ps1` targets only PIDs recorded for this project. It requests a
Windows console interrupt so the trainer can save `checkpoint_latest.pth`; a forced
stop is used only if that graceful request times out.

`cv_split.csv`, `splits_final.json`, environment/state files, run logs, predictions and weights are local
protected/generated artifacts and are intentionally ignored by Git.
