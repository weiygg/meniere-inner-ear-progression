# M0 external geometry reliability (locked benchmark)

Date: 2026-08-29

This is a fixed-model reliability analysis of the already exposed 50-person external
benchmark (100 ears; 300 AI-manual mask pairs). It is not a model-selection result
and cannot be used to tune M1-M3. Confidence intervals use 5,000 patient-clustered
bootstrap samples. ICC is two-way mixed, absolute-agreement, single-measure
ICC(A,1). The study team has not signed an acceptance threshold, so no feature is
labelled pass or fail.

| Structure/pair | Feature | ICC(A,1) (95% CI) | Bias (95% limits of agreement) | MAE | Mean relative error |
|---|---|---:|---:|---:|---:|
| SSC | volume, mm3 | 0.431 (0.239-0.588) | 1.980 (-2.560 to 6.520) | 2.576 | 21.2% |
| HSC | volume, mm3 | 0.734 (0.550-0.832) | 0.630 (-4.963 to 6.223) | 2.190 | 12.6% |
| PSC | volume, mm3 | 0.456 (0.302-0.576) | 2.403 (-2.836 to 7.642) | 3.120 | 22.6% |
| SSC | centerline length, mm | 0.549 (0.400-0.681) | -0.110 (-5.146 to 4.927) | 1.787 | 14.8% |
| HSC | centerline length, mm | 0.556 (0.358-0.732) | -0.960 (-5.149 to 3.230) | 1.631 | 13.1% |
| PSC | centerline length, mm | 0.469 (0.268-0.673) | -0.172 (-5.303 to 4.959) | 1.618 | 11.1% |
| SSC-HSC | unsigned plane angle, degrees | 0.525 (0.386-0.717) | -0.907 (-12.044 to 10.230) | 3.306 | 4.2% |
| SSC-PSC | unsigned plane angle, degrees | 0.091 (-0.018-0.531) | -3.005 (-19.464 to 13.454) | 3.863 | 4.4% |
| HSC-PSC | unsigned plane angle, degrees | 0.698 (0.545-0.798) | 0.127 (-4.287 to 4.541) | 1.680 | 2.0% |

Component counts and centerline voxel counts were also calculated. PSC component
count was constant and therefore has no estimable ICC. A PHI-safe machine-readable
aggregate is published in `data/manifests/m0_geometry_reliability_bootstrap5000.json`;
the complete paired tables remain in the Git-ignored run directory. Neither public
artifact contains case identifiers, paths, images, masks, or weights.

Interpretation: overlap/surface accuracy and geometry reliability answer different
questions. The continuous estimates above do not justify a blanket claim that all
derived canal geometry is reliable.
