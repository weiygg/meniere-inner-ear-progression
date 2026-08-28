# Protocol V2 complete execution runbook

This is the code-level reproduction guide for the Protocol V2 segmentation and
P-EBM project. It uses the five-epoch equal-budget pilot requested for M1/M2/M3.
It does not turn the historical internal benchmark or previously exposed external
set into an untouched external validation cohort.

## 1. Frozen scientific boundaries

- `LS_SEG_200` and `LS_CLIN_79` are different datasets. Identical local numeric
  names do not identify the same person; verified cross-dataset patient overlap is
  zero.
- All segmentation model selection uses only the locked `LS_SEG_200` development
  split: 280 training ears and 60 validation ears. The 60-ear historical internal
  benchmark is excluded from fingerprinting and model selection.
- M1, M2, and M3 use the same split and five epochs. M2 changes scanner-robust
  augmentation; M3 adds the prespecified soft-clDice term.
- The previously exposed external set may only be reported as a locked historical
  benchmark. It cannot select architecture, augmentation, loss, threshold, or
  postprocessing.
- A new confirmatory external cohort is evaluated once, only after all parameters
  are frozen. Its primary endpoint is macro Dice with a 95% confidence interval.
- Final clinical P-EBM remains blocked until the clinical crosswalk, signed
  codebook, affected/index-ear definition, audiometry timing, missing codes, and
  abnormal directions are resolved.

## 2. Software environment

Run from a Git checkout of branch `codex/protocol-v2-seg-pebm`. On Windows, use
ASCII-only locations for nnU-Net data and outputs because Blosc2 preprocessing was
observed to fail under a non-ASCII worktree path.

```powershell
$ProjectRoot = Resolve-Path .
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-segmentation.txt
python -m pytest -q
python scripts\audit_public_git.py --root $ProjectRoot
```

The exact direct dependencies are pinned in `requirements.txt` and
`requirements-segmentation.txt`. Every training run also writes Python, PyTorch,
GPU, trainer, split source, epoch budget, and external-label-loading status to its
`training_manifest.json`.

## 3. Local protected paths

Set these paths for the controlled local data. Never commit the referenced files.

```powershell
$LocalM0Manifest = '<LOCAL_M0_EAR_CROP_MANIFEST.csv>'
$NnRaw = '<ASCII_NNUNET_RAW_ROOT>'
$NnPreprocessed = '<ASCII_NNUNET_PREPROCESSED_ROOT>'
$NnResults = '<ASCII_NNUNET_RESULTS_ROOT>'
$RunRoot = '<GIT_IGNORED_LOCAL_RUN_ROOT>'
```

The local M0 manifest must contain `subject_id`, `side`, `split`, and `crop_path`.
Each `crop_path` points to a protected `.npz` ear crop containing `image` and the
three-channel `mask`. The frozen split manifest used by the audit must also contain
`subject_id`, `side` or `ear_side`, and `split`.

## 4. Registry and frozen-split gates

```powershell
python scripts\validate_dataset_registry.py --config-dir configs
python scripts\run_pipeline.py registry --config-dir configs --output-root $RunRoot
python scripts\audit_m0_split.py $LocalM0Manifest --output "$RunRoot\m0_split_provenance.json"
```

Expected registry result: `LS_SEG_200__LS_CLIN_79_patient_overlap = 0`. The split
audit must report patient-level separation with both ears of a person in the same
split.

## 5. Multiclass label audit and nnU-Net preparation

Audit label overlap before conversion, then apply the prespecified physical-space
nearest-exclusive policy.

```powershell
python scripts\audit_multiclass_label_overlap.py `
  --manifest $LocalM0Manifest `
  --output "$RunRoot\m1_multiclass_overlap_audit.json"

python scripts\prepare_nnunet_multiclass.py `
  --manifest $LocalM0Manifest `
  --nnunet-raw $NnRaw `
  --dataset-id 501 `
  --dataset-name LSSemicircularCanals `
  --spacing-mm 0.3472222 0.3472222 0.5 `
  --overlap-policy nearest-exclusive

python scripts\plan_nnunet_protocol_v2.py `
  --nnunet-raw $NnRaw `
  --nnunet-preprocessed $NnPreprocessed `
  --nnunet-results $NnResults `
  --dataset-id 501 `
  --dataset-name LSSemicircularCanals `
  --run-dir "$RunRoot\planning"
```

Planning is deliberately limited to one preprocessing worker (`-np 1`) on
Windows. The script copies the locked split into `splits_final.json` only after
successful integrity checking and records hashes of the fingerprint, plan, and
split.

## 6. Equal-budget M1, M2, and M3 training

Run the three models sequentially so the 6-GB GPU is not oversubscribed.

```powershell
python scripts\run_nnunet_protocol_v2.py M1 `
  --nnunet-raw $NnRaw --nnunet-preprocessed $NnPreprocessed `
  --nnunet-results $NnResults --dataset Dataset501_LSSemicircularCanals `
  --fold 0 --device cuda --num-epochs 5 --run-dir "$RunRoot\m1_pilot5"

python scripts\run_nnunet_protocol_v2.py M2 `
  --nnunet-raw $NnRaw --nnunet-preprocessed $NnPreprocessed `
  --nnunet-results $NnResults --dataset Dataset501_LSSemicircularCanals `
  --fold 0 --device cuda --num-epochs 5 --run-dir "$RunRoot\m2_pilot5"

python scripts\run_nnunet_protocol_v2.py M3 `
  --nnunet-raw $NnRaw --nnunet-preprocessed $NnPreprocessed `
  --nnunet-results $NnResults --dataset Dataset501_LSSemicircularCanals `
  --fold 0 --device cuda --num-epochs 5 --run-dir "$RunRoot\m3_pilot5"
```

`--continue-training` may be added only to resume the same experiment, split, and
epoch budget from its own checkpoint. Never resume one model from another model's
weights when comparing M1/M2/M3.

## 7. Patient-clustered validation summaries

nnU-Net output folders are trainer-specific. Define each model folder and produce
an aggregate-only summary with 5,000 patient bootstrap repetitions.

```powershell
$M1Model = Join-Path $NnResults 'Dataset501_LSSemicircularCanals\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_0'
$M2Model = Join-Path $NnResults 'Dataset501_LSSemicircularCanals\nnUNetTrainerProtocolV2M2__nnUNetPlans__3d_fullres\fold_0'
$M3Model = Join-Path $NnResults 'Dataset501_LSSemicircularCanals\nnUNetTrainerProtocolV2M3__nnUNetPlans__3d_fullres\fold_0'

python scripts\summarize_nnunet_validation.py `
  --summary "$M1Model\validation\summary.json" `
  --training-manifest "$RunRoot\m1_pilot5\training_manifest.json" `
  --checkpoint-final "$M1Model\checkpoint_final.pth" `
  --checkpoint-best "$M1Model\checkpoint_best.pth" `
  --output "$RunRoot\m1_pilot5_aggregate.json" --bootstrap 5000 --seed 20260828

python scripts\summarize_nnunet_validation.py `
  --summary "$M2Model\validation\summary.json" `
  --training-manifest "$RunRoot\m2_pilot5\training_manifest.json" `
  --checkpoint-final "$M2Model\checkpoint_final.pth" `
  --checkpoint-best "$M2Model\checkpoint_best.pth" `
  --output "$RunRoot\m2_pilot5_aggregate.json" --bootstrap 5000 --seed 20260828

python scripts\summarize_nnunet_validation.py `
  --summary "$M3Model\validation\summary.json" `
  --training-manifest "$RunRoot\m3_pilot5\training_manifest.json" `
  --checkpoint-final "$M3Model\checkpoint_final.pth" `
  --checkpoint-best "$M3Model\checkpoint_best.pth" `
  --output "$RunRoot\m3_pilot5_aggregate.json" --bootstrap 5000 --seed 20260828
```

Compare M1/M2/M3 only on the locked internal validation split. A five-epoch pilot
is an implementation and screening result, not a final model-performance claim.

## 8. Frozen exposed-external evaluation

Run this section only after selecting M1/M2/M3 from internal validation. The
external study-QC tables contain the frozen union-localizer centers recorded before
nnU-Net selection. Manual-mask directory names define the two prespecified 25-person
evaluation subsets but masks are not loaded during crop preparation or inference.

```powershell
$ExternalStudyQc = '<LOCAL_FROZEN_EXTERNAL_STUDY_QC.csv>'
$ExternalOverrideQc = '<LOCAL_CENTER3_OVERRIDE_STUDY_QC.csv>'
$ManualCenter2 = '<LOCAL_CENTER2_MANUAL_REFERENCE_DIR>'
$ManualCenter3 = '<LOCAL_CENTER3_MANUAL_REFERENCE_DIR>'
$SelectedModelFolder = '<LOCAL_SELECTED_NNUNET_MODEL_FOLDER>'
$ExternalRun = Join-Path $RunRoot 'selected_model_exposed_external'

python scripts\prepare_nnunet_external_crops.py `
  --center2-study-qc $ExternalStudyQc --center3-study-qc $ExternalStudyQc `
  --manual-center2-dir $ManualCenter2 --manual-center3-dir $ManualCenter3 `
  --override-center3-study-qc $ExternalOverrideQc `
  --output-dir "$ExternalRun\prepared"

python scripts\predict_nnunet_protocol_v2.py `
  --input-dir "$ExternalRun\prepared\imagesTs" `
  --output-dir "$ExternalRun\predicted_crops" `
  --model-folder $SelectedModelFolder --fold 0 `
  --checkpoint checkpoint_final.pth --device cuda

python scripts\restore_nnunet_external_predictions.py `
  --prediction-dir "$ExternalRun\predicted_crops" `
  --metadata "$ExternalRun\prepared\external_crop_metadata.json" `
  --output-dir "$ExternalRun\restored"

python scripts\evaluate_nnunet_external_manual_masks.py `
  --manual-center2-dir $ManualCenter2 --manual-center3-dir $ManualCenter3 `
  --manifest-center2 "$ExternalRun\restored\c2_prediction_manifest.csv" `
  --manifest-center3 "$ExternalRun\restored\c3_prediction_manifest.csv" `
  --checkpoint "$SelectedModelFolder\fold_0\checkpoint_final.pth" `
  --output-dir "$ExternalRun\evaluation" --bootstrap 5000 --seed 20260829
```

The preparation metadata must report `external_labels_loaded: false`. Patient-level
predictions and metrics remain local; only reviewed aggregate summaries may enter
Git. Center 2 and Center 3 are strata from one external institution and have already
been exposed, so they cannot support a new independent confirmatory claim.

## 9. Geometry reliability

After blinded manual masks are available, compute physical-space geometry
reliability locally. The input and generated patient-level CSV/figures remain
Git-ignored.

```powershell
$ProtectedGeometryMetrics = '<LOCAL_PAIRED_MANUAL_AI_MASK_METRICS.csv>'
python scripts\evaluate_geometry_reliability.py `
  $ProtectedGeometryMetrics "$RunRoot\geometry_reliability" `
  --config configs\segmentation_experiments.yaml `
  --bootstrap 5000 --seed 20260826
```

Report ICC(A,1), 95% patient-bootstrap confidence intervals, Bland-Altman limits,
MAE, and relative error. Do not invent a pass threshold: the current protocol
records that threshold as unsigned and therefore blocked.

## 10. Clinical P-EBM gate

The following commands audit readiness without fitting a final clinical model.

```powershell
python scripts\validate_pebm_eligibility.py `
  --schema data\manifests\clinical_feature_schema.json `
  --codebook configs\clinical_codebook.yaml `
  --output "$RunRoot\pebm_eligibility.json"

python scripts\run_pipeline.py clinical_qc `
  --config-dir configs --output-root $RunRoot --seed 20260826

python scripts\run_pipeline.py pebm_eligibility `
  --config-dir configs --output-root $RunRoot --seed 20260826
```

The `pebm_eligibility` phase currently returns exit code 2 and records its blockers;
that is the expected safe stop until the codebook is signed. Do not infer missing
crosswalks, endpoints, units, ear assignments, or visit timing from numeric filenames.

## 11. Code snapshot and verification

Commit the reviewed code first. Then create a text-only code snapshot. The builder
uses `git ls-files`, refuses dirty tracked files, applies a strict allowlist, rejects
images, masks, weights, spreadsheets, CSVs, archives, local paths, credentials, and
members larger than 5 MiB, and writes an internal per-file SHA-256 inventory.

```powershell
$Deliverables = '<LOCAL_CODE_ARCHIVE_OUTPUT_DIR>'
python scripts\audit_public_git.py --root $ProjectRoot
python -m pytest -q
python scripts\build_protocol_v2_code_bundle.py `
  --root $ProjectRoot --output-dir $Deliverables
```

Verify a saved archive against its sidecar:

```powershell
$Archive = '<PATH_TO_CODE_ZIP>'
$Recorded = (Get-Content "$Archive.sha256.json" | ConvertFrom-Json).sha256
$Observed = (Get-FileHash $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($Recorded -ne $Observed) { throw 'Code archive SHA-256 mismatch' }
```

The Git commit or release tag is the authoritative version identifier. The ZIP is
a portable copy of the same tracked code and contains `BUNDLE_METADATA.json` with
the commit, runtime versions, privacy boundary, excluded-file report, and SHA-256
for every included member.
