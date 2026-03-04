"""VAR(1) data generator with multivariate-t innovations.

Produces synthetic returns that mimic heavy-tailed, cross-correlated financial
data without requiring real market downloads.

Usage::

    from streamlit_test.var_generate import simulate_var1
    df = simulate_var1(n_obs=17520, d=3, seed=42)
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def _default_A(d: int) -> np.ndarray:
    """Diagonal ~0.3 with off-diagonal spillovers ~0.1."""
    A = np.full((d, d), 0.1)
    np.fill_diagonal(A, 0.3)
    return A


def _default_Sigma(d: int) -> np.ndarray:
    """Scaled covariance matrix with moderate cross-correlation."""
    scale = 1e-4
    rho = 0.4
    Sigma = np.full((d, d), rho * scale)
    np.fill_diagonal(Sigma, scale)
    return Sigma


def simulate_var1(
    n_obs: int = 17_520,
    d: int = 3,
    A: Optional[np.ndarray] = None,
    Sigma: Optional[np.ndarray] = None,
    df: float = 3.0,
    seed: int = 42,
    symbols: Optional[Sequence[str]] = None,
    start: str = "2022-01-01",
    freq: str = "h",
) -> pd.DataFrame:
    """Simulate a d-dimensional VAR(1) process with multivariate-t innovations.

    Model: r_t = A @ r_{t-1} + eps_t
    where  eps_t = sqrt(df / chi2) * L @ z,  chi2 ~ chi2(df),  z ~ N(0, I)

    Parameters
    ----------
    n_obs : number of observations (default 17 520 = 2 years hourly)
    d : number of assets
    A : (d, d) autoregressive coefficient matrix.  Defaults to diag(0.3) + 0.1.
    Sigma : (d, d) innovation scale matrix.  Defaults to 1e-4 * (I + 0.4).
    df : degrees of freedom for the multivariate-t (lower = heavier tails)
    seed : random seed
    symbols : column names (default VAR-1 .. VAR-d)
    start : start date for the hourly index
    freq : pandas frequency alias

    Returns
    -------
    pd.DataFrame with shape (n_obs, d), hourly DatetimeIndex, columns = symbols
    """
    rng = np.random.default_rng(seed)

    if A is None:
        A = _default_A(d)
    if Sigma is None:
        Sigma = _default_Sigma(d)
    if symbols is None:
        symbols = [f"VAR-{i + 1}" for i in range(d)]

    A = np.asarray(A, dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)

    # Stationarity check
    eigvals = np.abs(np.linalg.eigvals(A))
    spectral_radius = eigvals.max()
    if spectral_radius >= 1.0:
        raise ValueError(
            f"VAR(1) is non-stationary: spectral radius of A = {spectral_radius:.4f} >= 1"
        )

    L = np.linalg.cholesky(Sigma)

    # Pre-generate all random draws
    Z = rng.standard_normal((n_obs, d))
    chi2 = rng.chisquare(df, size=n_obs)
    scaling = np.sqrt(df / chi2)  # (n_obs,)

    # Multivariate-t innovations: eps_t = sqrt(df/chi2_t) * L @ z_t
    eps = scaling[:, None] * (Z @ L.T)

    # VAR(1) recursion
    R = np.zeros((n_obs, d), dtype=np.float64)
    R[0] = eps[0]
    for t in range(1, n_obs):
        R[t] = A @ R[t - 1] + eps[t]

    index = pd.date_range(start=start, periods=n_obs, freq=freq)
    return pd.DataFrame(R, index=index, columns=list(symbols))
