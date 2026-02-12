"""Preprocessing pipeline for Laplace margins.

Reuses GARCH fitting from cascades.preprocess, then applies PIT to Laplace
margins instead of exponential margins.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from cascades.utils import EmpiricalCDF


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Forward to cascades.preprocess lazily.

    This keeps module import lightweight so users of laplace_* utilities do not
    require optional GARCH dependencies at import time.
    """
    from cascades.preprocess import compute_log_returns as _compute_log_returns

    return _compute_log_returns(prices)


def fit_garch(
    df: pd.DataFrame,
    dist: str = "t",
    return_filter: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame] | Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Forward to cascades.preprocess lazily."""
    from cascades.preprocess import fit_garch as _fit_garch

    return _fit_garch(df, dist=dist, return_filter=return_filter)


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
