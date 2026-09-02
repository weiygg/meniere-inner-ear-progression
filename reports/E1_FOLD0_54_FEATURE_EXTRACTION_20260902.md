# E1 fold 0 (54 epoch) predicted-mask feature extraction

## Scope and provenance

- Model: formal E1 fold 0 fixed 54-epoch checkpoint.
- Source: frozen predictions from the one-time exposed external evaluation completed on 2026-09-02.
- Cohorts: Center 2 and Center 3 are two previously exposed subcohorts from the same institution. They are not two independent hospitals and are not a new confirmatory external validation.
- These are morphology and centerline geometry features derived from predicted masks. They are not manual-mask ground truth and do not include image-intensity or PyRadiomics texture features.

## Completion and QC

| Item | Result |
|---|---:|
| Patients | 50 |
| Ears | 100 |
| Predicted canal masks / basic-feature rows | 300 |
| Centerline rows passing QC | 299 / 300 |
| Plane-angle rows | 298 |
| Bilateral-asymmetry rows | 450 |
| Feature-extraction errors | 1 |

The single error was an HSC centerline whose skeleton contained fewer than four voxels. Its basic shape features remain available; its centerline features and two HSC-dependent plane angles are unavailable. The public aggregate contains no study identifier or path.

## Selected pooled descriptive results

| Structure | Volume, mean (mm3) | Surface area, mean (mm2) | Sphericity, mean | Centerline length, mean (mm) | Mean diameter (mm) | Mean curvature (1/mm) |
|---|---:|---:|---:|---:|---:|---:|
| SSC | 16.345 | 60.522 | 0.515 | 11.789 | 1.024 | 0.293 |
| HSC | 21.574 | 63.981 | 0.587 | 9.328 | 1.282 | 0.391 |
| PSC | 17.335 | 64.701 | 0.501 | 13.303 | 0.988 | 0.271 |

Pooled acute plane angles were 84.284 degrees for HSC-PSC (n=99), 83.612 degrees for HSC-SSC (n=99), and 84.919 degrees for PSC-SSC (n=100).

## Deliverables and analysis boundary

- Patient-level results are kept locally in the generated Excel workbook and local CSV/JSON outputs; they are intentionally excluded from GitHub.
- The public JSON is aggregate-only and reports center-specific and pooled distributions (n, mean, SD, median, Q1, Q3) for shape, centerline, plane-angle, and bilateral-asymmetry features.
- Before clinical modeling, predicted masks require visual segmentation QC. Failed centerlines must be excluded from centerline and plane analyses.
- These exposed subcohorts must not be used for hyperparameter tuning or model selection.

## Reproducible commands

```powershell
python src\29_extract_predicted_canal_features.py `
  --mask-root results\runs\e1_fold0_54_exposed_external_20260902\restored\predicted_masks `
  --output-dir results\runs\e1_fold0_54_exposed_external_20260902\features_v2_center_labeled `
  --center2-manifest results\runs\e1_fold0_54_exposed_external_20260902\restored\c2_prediction_manifest.csv `
  --center3-manifest results\runs\e1_fold0_54_exposed_external_20260902\restored\c3_prediction_manifest.csv

python scripts\summarize_predicted_canal_features.py `
  --feature-dir results\runs\e1_fold0_54_exposed_external_20260902\features_v2_center_labeled `
  --output-json data\manifests\e1_fold0_54_feature_extraction_aggregate_20260902.json `
  --model-label formal_E1_fold0_54epoch_fixed_checkpoint
```

