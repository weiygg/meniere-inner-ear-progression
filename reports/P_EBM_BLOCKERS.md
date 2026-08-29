# P-EBM blockers

Final `primary_common_core_pebm` has not been run.

The 2026-08-29 schema-only eligibility audit evaluated 73 variable rows. Zero
primary events were eligible and all 73 rows remained blocked because required
definitions were not frozen. The phase runner independently returned
`clinical_codebook_not_signed`. These are readiness findings, not a fitted P-EBM
result.

Unresolved gates:

1. signed clinical codebook with units, coding, missing codes, and abnormal direction;
2. explicit Z2 patient/visit crosswalk;
3. confirmation that audiometry values are dB HL and a signed timing window;
4. exact abnormal-hearing features and thresholds;
5. bilateral abnormal and first-affected/index-ear rules;
6. cross-center compatible common-core variables and supported mixture types.

The existing aggregate hearing-rule sensitivity analysis is retained but cannot be
used to choose the final rule by yield. Previous P-EBM event sequences and stages are
legacy evidence only.
