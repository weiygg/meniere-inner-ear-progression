# Configurations

Commit de-identified examples only. Real paths and protected-data settings belong in
`configs/analysis.yaml`, which is ignored by Git.

Protocol V2 separates the authoritative configuration into dataset registry, overlap,
segmentation, clinical progression, codebook, and P-EBM files. The legacy
`center_split.yaml` remains as a compatibility layer; it is no longer the only source
of study-role truth.
