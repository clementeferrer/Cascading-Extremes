"""Preprocessing pipeline for Laplace margins.

Reuses GARCH fitting from cascades.preprocess, then applies PIT to Laplace
margins instead of exponential margins.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from cascades.preprocess import compute_log_returns, fit_garch  # noqa: F401
from cascades.utils import EmpiricalCDF


def laplace_quantile(u: np.ndarray) -> np.ndarray:
    """Inverse Laplace CDF: F_Lap^{-1}(u) = -sign(u-0.5) * log(1 - 2|u-0.5|)."""
    centered = u - 0.5
    return -np.sign(centered) * np.log(np.maximum(1.0 - 2.0 * np.abs(centered), 1e-15))


def laplace_cdf(x: np.ndarray) -> np.ndarray:
    """Laplace CDF: F_Lap(x) = 0.5 + 0.5 * sign(x) * (1 - exp(-|x|))."""
    return 0.5 + 0.5 * np.sign(x) * (1.0 - np.exp(-np.abs(x)))


def standardize_laplace(
    residuals: pd.DataFrame,
    pit_clip: float = 1e-6,
) -> Tuple[pd.DataFrame, Dict[str, EmpiricalCDF]]:
    """Transform GARCH residuals to standard Laplace margins via PIT.

    Returns
    -------
    X_df : pd.DataFrame
        Values in R (not restricted to positive).
    cdfs : dict
        Empirical CDF per column for inversion.
    """
    X = {}
    cdfs: Dict[str, EmpiricalCDF] = {}

    for col in residuals.columns:
        series = residuals[col].dropna()
        sorted_vals = np.sort(series.values)
        cdf = EmpiricalCDF(sorted_values=sorted_vals, eps=pit_clip)
        u = cdf.cdf(series.values)
        x = laplace_quantile(u)
        X[col] = pd.Series(x, index=series.index)
        cdfs[col] = cdf

    X_df = pd.DataFrame(X)
    n = max(len(X_df), 2)
    X_df = X_df / np.log(n / 2)
    return X_df, cdfs


def inverse_standardize_laplace(
    X: pd.DataFrame,
    cdfs: Dict[str, EmpiricalCDF],
) -> pd.DataFrame:
    """Invert Laplace standardization back to GARCH residuals."""
    n = max(len(X), 2)
    y = X * np.log(n / 2)
    residuals = {}
    for col in X.columns:
        series = y[col].dropna()
        u = laplace_cdf(series.values)
        u = np.clip(u, 1e-6, 1.0 - 1e-6)
        resid = cdfs[col].ppf(u)
        residuals[col] = pd.Series(resid, index=series.index)
    return pd.DataFrame(residuals)
