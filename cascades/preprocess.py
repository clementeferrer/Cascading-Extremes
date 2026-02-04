from typing import Dict, Tuple

import numpy as np
import pandas as pd
from arch import arch_model

from cascades.utils import EmpiricalCDF


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    logp = np.log(prices)
    returns = logp.diff().dropna(how="all")
    return returns


def fit_garch(df: pd.DataFrame, dist: str = "t") -> Tuple[pd.DataFrame, pd.DataFrame]:
    residuals = {}
    sigma = {}

    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue
        am = arch_model(series * 100.0, vol="GARCH", p=1, q=1, dist=dist)
        res = am.fit(disp="off")
        cond_vol = res.conditional_volatility / 100.0
        resid = res.resid / 100.0
        aligned = series.loc[cond_vol.index]
        resid = resid.loc[aligned.index]
        cond_vol = cond_vol.loc[aligned.index]
        residuals[col] = resid
        sigma[col] = cond_vol

    residuals_df = pd.DataFrame(residuals)
    sigma_df = pd.DataFrame(sigma)
    return residuals_df, sigma_df


def standardize(residuals: pd.DataFrame, pit_clip: float = 1.0e-6) -> Tuple[pd.DataFrame, Dict[str, EmpiricalCDF]]:
    X = {}
    cdfs: Dict[str, EmpiricalCDF] = {}

    for col in residuals.columns:
        series = residuals[col].dropna()
        sorted_vals = np.sort(series.values)
        cdf = EmpiricalCDF(sorted_values=sorted_vals, eps=pit_clip)
        u = cdf.cdf(series.values)
        y = -np.log(1.0 - u)
        X[col] = pd.Series(y, index=series.index)
        cdfs[col] = cdf

    X_df = pd.DataFrame(X)
    n = max(len(X_df), 2)
    X_df = X_df / np.log(n)
    return X_df, cdfs


def inverse_standardize(X: pd.DataFrame, cdfs: Dict[str, EmpiricalCDF]) -> pd.DataFrame:
    n = max(len(X), 2)
    y = X * np.log(n)
    residuals = {}
    for col in X.columns:
        series = y[col].dropna()
        u = 1.0 - np.exp(-series.values)
        u = np.clip(u, 1.0e-6, 1.0 - 1.0e-6)
        resid = cdfs[col].ppf(u)
        residuals[col] = pd.Series(resid, index=series.index)
    return pd.DataFrame(residuals)
