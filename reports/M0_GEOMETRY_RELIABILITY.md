# M0 external geometry reliability (locked benchmark)

Date: 2026-08-26

This is a fixed-model reliability analysis of the already exposed 50-person external
benchmark (100 ears; 300 AI-manual mask pairs). It is not a model-selection result
and cannot be used to tune M1-M3. Confidence intervals use 1,000 patient-clustered
bootstrap samples. ICC is two-way mixed, absolute-agreement, single-measure
ICC(A,1). The study team has not signed an acceptance threshold, so no feature is
labelled pass or fail.

| Structure/pair | Feature | ICC(A,1) (95% CI) | Bias | MAE | Mean relative error |
|---|---|---:|---:|---:|---:|
| SSC | volume, mm3 | 0.431 (0.240-0.596) | 1.980 | 2.576 | 21.2% |
| HSC | volume, mm3 | 0.734 (0.549-0.832) | 0.630 | 2.190 | 12.6% |
| PSC | volume, mm3 | 0.456 (0.302-0.576) | 2.403 | 3.120 | 22.6% |
| SSC | centerline length, mm | 0.549 (0.406-0.675) | -0.110 | 1.787 | 14.8% |
| HSC | centerline length, mm | 0.556 (0.349-0.727) | -0.960 | 1.631 | 13.1% |
| PSC | centerline length, mm | 0.469 (0.270-0.669) | -0.172 | 1.618 | 11.1% |
| SSC-HSC | unsigned plane angle, degrees | 0.525 (0.373-0.715) | -0.907 | 3.306 | 4.2% |
| SSC-PSC | unsigned plane angle, degrees | 0.091 (-0.020-0.530) | -3.005 | 3.863 | 4.4% |
| HSC-PSC | unsigned plane angle, degrees | 0.698 (0.545-0.802) | 0.127 | 1.680 | 2.0% |

Component counts and centerline voxel counts were also calculated. PSC component
count was constant and therefore has no estimable ICC. The full machine-readable
summary remains in the Git-ignored run directory; this report contains only
aggregate values and no case identifiers, paths, images, masks, or weights.

Interpretation: overlap/surface accuracy and geometry reliability answer different
questions. The continuous estimates above do not justify a blanket claim that all
derived canal geometry is reliable.
