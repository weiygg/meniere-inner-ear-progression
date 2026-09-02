# Segmentation V2 final report

Status: **IN PROGRESS — formal E1 five-fold training paused after fold 0**

| Model | SSC | HSC | PSC | Macro Dice | Surface Dice | ASSD | HD95 | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| V1 baseline internal | 0.794 | 0.837 | 0.810 | 0.8136 | not available | 0.092 | 0.492 | COMPLETED |
| V1 exposed external | 0.655 | 0.712 | 0.639 | 0.6688 | 0.975 | 0.207 | 0.797 | COMPLETED |
| nnU-Net fold-0 five-epoch pilot M1 | 0.781 | 0.817 | 0.791 | 0.7964 | not pooled here | not pooled here | not pooled here | COMPLETED, not OOF |
| nnU-Net fold-0 bias-field pilot M2 | 0.785 | 0.817 | 0.794 | 0.7984 | not pooled here | not pooled here | not pooled here | COMPLETED, not OOF |
| Formal E1 fold 0 (54 epochs) | 0.796 | 0.836 | 0.814 | 0.8153 | 0.984 at 0.5 mm | 0.111 | 0.432 | COMPLETED, single internal fold |
| Formal E1 fold 0 exposed external | 0.682 | 0.718 | 0.645 | 0.6817 | 0.988 at 1 mm | 0.208 | 0.643 | COMPLETED, exposed benchmark only |
| Residual nnU-Net |  |  |  |  |  |  |  | GPU PROBE PASSED; OOF NOT RUN |
| High-depth-resolution nnU-Net |  |  |  |  |  |  |  | GPU PROBE PASSED; OOF NOT RUN |
| V2 final OOF |  |  |  |  |  |  |  | PAUSED: fold 0 complete; folds 1-4 incomplete |
| V2 frozen external |  |  |  |  |  |  |  | BLOCKED: model not frozen |

The fold-0 pilots used one 30-patient validation set and must not be labelled
five-fold OOF results. No paired improvement waterfall is reported until matching
OOF ablations exist.

## Execution status

- **COMPLETED:** V1 audit; volume-shift tests; fixed 200-patient/400-ear five-fold
  split; Dataset502 4-class construction; integrity validation; preprocessing;
  plain/residual/high-depth-resolution planning; CUDA forward/backward probes;
  E5/E6 loss gradient smoke tests; frozen-external gate test; 30-patient V1
  exposed-external top/bottom failure selection with 60 local tri-planar overlays.
- **PAUSED:** formal E1 uses a fixed 54-epoch compute cap while retaining the
  native 1000-epoch nnU-Net learning-rate schedule horizon, fixed
  patient-level folds 0-4 in serial order, native CUDA mixed precision and a
  one-epoch checkpoint interval. Fold 0 completed on 2026-09-02 with internal
  Macro Dice 0.8153 (40 patients/80 ears). Fold 1 was paused during epoch 0 on
  2026-09-02 for a user-requested rapid exposed-external benchmark; folds 1-4 are
  incomplete. This single-fold estimate is not the final five-fold OOF result.
  The detached worker was launched on 2026-08-31;
  live PIDs, logs and checkpoints remain local and Git-ignored.
- **NOT RUN:** formal E2/E4/E5/E6 five-fold training. Stage B is considered only
  after the E1 OOF summary is available and is triggered by an internal OOF macro
  Dice below 0.83, never by external performance.
- **BLOCKED:** final five-fold model selection, `MODEL_FREEZE.md`, a new
  confirmatory V2 external evaluation, paired OOF ablations, internal OOF failure
  cases and morphology-fidelity comparisons. The previously exposed external
  benchmark is complete but cannot resolve these gates.

The high-depth-resolution plan changes one depth stride from 2 to 1, preserving a
12-voxel depth at the bottleneck instead of 6 while leaving the original plan
untouched. Batch-1 mixed-precision forward/backward used about 1.01 GiB allocated
GPU memory. Residual Encoder nnU-Net also passed at batch 1 (about 1.55 GiB
allocated); its planner-suggested batch 6 is not assumed safe on 6 GB VRAM.

Current answers:

- Q1/Q2: V1 external SSC and PSC show volume overprediction and lower precision
  than recall, superimposed on strong 1-mm surface agreement. Boundary thickness
  and endpoint convention are plausible contributors, not proven sole causes.
- Q3: TinyViT leaves six z tokens at the bottleneck; this is a plausible limitation,
  not a demonstrated cause. A non-destructive 12-depth high-resolution plan now
  exists and passed CUDA probing, but paired OOF training is still required.
- Q4-Q8: not answerable from the current equal-budget single-fold pilots.
- Q9: formal internal OOF >=0.83 has not been tested.
- Q10: no frozen V2 external result exists; the historical exposed result is <0.78.

The V1 external masks are larger than their references for SSC and PSC (median
prediction/reference ratios 1.162 and 1.187) and have lower precision than recall.
Together with 1-mm surface Dice 0.958-0.996 and submillimetre aggregate HD95, this
supports boundary thickness and endpoint convention as leading explanations. It
does not prove that scanner shift, contrast, motion or annotation policy are absent.

The zoomed tri-planar overlays confirm that low-Dice cases can remain correctly
localized while dashed prediction contours extend beyond solid references in canal
thickness and terminal segments; the top cases show much closer boundary and
endpoint agreement. Formal motion/contrast/FOV categories were not assigned because
the available metrics table has no blinded radiologist quality ratings. A second
observer/repeat-annotation analysis is **recommended but not executable with the
currently available data**; no second-reader masks were found or fabricated.

The two external subsets are two prespecified batches from the same external
institution and are already exposed. They are not two independent hospitals and
cannot serve as a new confirmatory validation after V2 development.

See `SEGMENTATION_V2_BASELINE_AUDIT.md` for the complete verified baseline.
