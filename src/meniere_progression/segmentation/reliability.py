from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReliabilityEstimate:
    icc_a1: float
    bland_altman_bias: float
    bland_altman_lower: float
    bland_altman_upper: float
    mean_absolute_error: float
    mean_relative_error: float


def icc_a1(reference: np.ndarray, automatic: np.ndarray) -> float:
    """Two-way mixed-effects, absolute-agreement, single-measure ICC(A,1)."""
    reference = np.asarray(reference, dtype=float)
    automatic = np.asarray(automatic, dtype=float)
    if reference.shape != automatic.shape or reference.ndim != 1 or len(reference) < 2:
        raise ValueError("ICC requires paired one-dimensional arrays with at least two rows")
    matrix = np.column_stack([reference, automatic])
    n, k = matrix.shape
    grand = matrix.mean()
    row_means = matrix.mean(axis=1)
    column_means = matrix.mean(axis=0)
    ms_rows = k * np.sum((row_means - grand) ** 2) / (n - 1)
    ms_columns = n * np.sum((column_means - grand) ** 2) / (k - 1)
    residual = matrix - row_means[:, None] - column_means[None, :] + grand
    ms_error = np.sum(residual**2) / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    return float((ms_rows - ms_error) / denominator) if denominator else float("nan")


def paired_reliability(reference: np.ndarray, automatic: np.ndarray) -> ReliabilityEstimate:
    reference = np.asarray(reference, dtype=float)
    automatic = np.asarray(automatic, dtype=float)
    if reference.shape != automatic.shape:
        raise ValueError("Paired arrays must have identical shapes")
    difference = automatic - reference
    bias = float(difference.mean())
    sd = float(difference.std(ddof=1)) if len(difference) > 1 else 0.0
    denominator = np.maximum(np.abs(reference), np.finfo(float).eps)
    return ReliabilityEstimate(
        icc_a1=icc_a1(reference, automatic),
        bland_altman_bias=bias,
        bland_altman_lower=bias - 1.96 * sd,
        bland_altman_upper=bias + 1.96 * sd,
        mean_absolute_error=float(np.abs(difference).mean()),
        mean_relative_error=float((np.abs(difference) / denominator).mean()),
    )
