"""Computation helpers — wraps second_phase model inference without duplication."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import ks_2samp, wasserstein_distance

from second_phase.dataset import spherical_features, build_token_sphere, build_tokens_from_arrays
from second_phase.extremes import ConstantThreshold, compute_global_threshold as _compute_global_threshold
from second_phase.genealogy import (
    Genealogy,
    build_genealogy,
    compute_parent_probabilities,
)
from second_phase.simulate import autoregressive_generate, prompt_generate, generate_with_genealogy

from streamlit_test.data_loader import AppData


# ── Hawkes decomposition ─────────────────────────────────────────────────


def compute_hawkes_decomposition(
    data: AppData,
    T: np.ndarray,
    R: np.ndarray,
    W: np.ndarray,
    dT: np.ndarray,
    window: int = 128,
    tokens: np.ndarray | None = None,
    q_model_override=None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute Hawkes intensity decomposition over a trailing window.

    Returns (lam, psi, kernel) or (None, None, None) if too few events.
    """
    model = data.model
    if len(T) < 3:
        return None, None, None

    max_len = getattr(model.cfg, "max_len", window)
    window = min(window, max_len, len(T))
    start = max(0, len(T) - window)

    T_w = T[start:]
    R_w = R[start:]
    W_w = W[start:]
    dT_w = dT[start:]

    d_input = model.input_proj.in_features
    if tokens is not None:
        tok = tokens[start:]
    else:
        # Compute thresholds for enriched token features
        u_w_thresh = None
        if d_input > 14:
            q = q_model_override if q_model_override is not None else data.q_model
            with torch.no_grad():
                u_w_thresh = q(torch.tensor(W_w, dtype=torch.float32)).numpy()
        tok = build_tokens_from_arrays(W_w, R_w, dT_w, u=u_w_thresh)

    # Truncate to model's expected d_input (backward compat)
    if tok.shape[1] > d_input:
        tok = tok[:, :d_input]

    tokens_t = torch.tensor(tok[None, :, :], dtype=torch.float32)
    T_tensor = torch.tensor(T_w[None, :], dtype=torch.float32)
    R_tensor = torch.tensor(R_w[None, :], dtype=torch.float32)
    W_tensor = torch.tensor(W_w[None, :, :], dtype=torch.float32)
    log_r = torch.log(R_tensor + 1e-8)

    with torch.no_grad():
        lam, psi, kernel = model.hawkes_intensity(
            model.encode(tokens_t), T_tensor, log_r, W=W_tensor, return_kernel=True,
        )

    return (
        lam.squeeze(0).numpy(),
        psi.squeeze(0).numpy(),
        kernel.squeeze(0).numpy(),
    )


def compute_full_intensity(
    data: AppData,
    T: np.ndarray,
    R: np.ndarray,
    W: np.ndarray,
    dT: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute intensity for all events using overlapping chunks.

    Returns (lam, psi, kernel) for the full sequence (may be approximate for
    very long sequences due to windowing).
    """
    model = data.model
    max_len = getattr(model.cfg, "max_len", 256)

    if len(T) <= max_len:
        lam, psi, kernel = compute_hawkes_decomposition(data, T, R, W, dT, window=max_len)
        return lam, psi, kernel

    # For long sequences, compute in the largest window we can
    return compute_hawkes_decomposition(data, T, R, W, dT, window=max_len)


# ── Genealogy ────────────────────────────────────────────────────────────


def build_genealogy_from_hawkes(
    lam: np.ndarray,
    psi: np.ndarray,
    kernel: np.ndarray,
) -> Genealogy:
    """Build genealogy from Hawkes decomposition arrays."""
    return build_genealogy(lam, psi, kernel)


def compute_parent_probs(
    lam: np.ndarray,
    psi: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    """Compute (n, n+1) parent probability matrix."""
    mu = np.maximum(lam - psi, 0.0)
    return compute_parent_probabilities(lam, mu, kernel)


# ── Returns conversion ───────────────────────────────────────────────────


def _laplace_quantile(u: np.ndarray) -> np.ndarray:
    centered = u - 0.5
    return -np.sign(centered) * np.log(np.maximum(1.0 - 2.0 * np.abs(centered), 1e-15))


def returns_to_laplace(
    pct_returns: Dict[str, float],
    cdfs_data: Dict[str, np.ndarray],
    asset_order: list,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Convert percentage returns to Laplace margins.

    Returns (X, R, W) where X is the Laplace vector, R = ||X||_2, W = X/R.
    """
    from cascades.utils import EmpiricalCDF

    X_vals = []
    for asset in asset_order:
        sorted_vals = cdfs_data[asset]
        cdf = EmpiricalCDF(sorted_values=sorted_vals, eps=1e-6)
        r_decimal = pct_returns[asset] / 100.0
        u = cdf.cdf(np.array([r_decimal]))[0]
        x = _laplace_quantile(np.array([u]))[0]
        X_vals.append(x)

    X = np.array(X_vals, dtype=np.float32)
    n = max(len(cdfs_data[asset_order[0]]), 2)
    X = X / np.log(n / 2)

    R = float(np.linalg.norm(X))
    if R < 1e-8:
        return X, R, np.zeros_like(X)
    W = X / R
    return X, R, W


compute_global_threshold = _compute_global_threshold


def check_extremality(
    W: np.ndarray,
    R: float,
    q_model,
    u_override: float | None = None,
) -> Tuple[bool, float]:
    """Check if (R, W) exceeds the threshold.

    When *u_override* is given the directional q_model is skipped and the
    scalar is used directly (global threshold mode).

    Returns (is_extreme, threshold).
    """
    if u_override is not None:
        return R > u_override, u_override
    W_t = torch.tensor(W[None, :], dtype=torch.float32)
    with torch.no_grad():
        u_tau = q_model(W_t).item()
    return R > u_tau, u_tau


# ── Generation ───────────────────────────────────────────────────────────


def build_real_context_prompt(real_events: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Build prompt context from real events for autoregressive continuation.

    Shifts T so last real event is at t=0, matching the web API pattern
    used by /generate/from-returns.
    """
    T = real_events["T"].astype(np.float32)
    T_shifted = T - float(T[-1])  # last event at t=0
    dT = np.diff(T, prepend=T[0]).astype(np.float32)
    if len(dT) > 1:
        med = float(np.median(dT[1:]))
        dT[0] = med if np.isfinite(med) and med > 1e-6 else 1.0
    elif len(dT) == 1:
        dT[0] = 1.0
    return {
        "W": real_events["W"].astype(np.float32),
        "R": real_events["R"].astype(np.float32),
        "T": T_shifted,
        "dT": dT,
    }


def run_autoregressive_generation(
    w0: np.ndarray,
    r0: float,
    max_time: float,
    data: AppData,
    temperature: float = 1.0,
    prompt: Optional[Dict[str, np.ndarray]] = None,
    q_model_override=None,
) -> Dict[str, np.ndarray]:
    """Run autoregressive generation (LLM-style)."""
    q = q_model_override if q_model_override is not None else data.q_model
    return autoregressive_generate(
        w0, r0, max_time, data.model, q,
        temperature=temperature, prompt=prompt,
    )


def run_generation_with_genealogy(
    data: AppData,
    asset_idx: int,
    magnitude: float,
    horizon: float,
) -> Tuple[Dict[str, np.ndarray], Genealogy]:
    """Run Ogata thinning generation with full genealogy."""
    max_events = data.cfg["simulate"]["horizon_events"]
    safety_factor = data.cfg["simulate"].get("safety_factor", 1.5)
    events, genealogy = prompt_generate(
        asset_idx, magnitude, horizon,
        max_events=max_events,
        model=data.model,
        q_model=data.q_model,
        safety_factor=safety_factor,
    )
    return events, genealogy


def run_generation_with_genealogy_w0(
    data: AppData,
    w0: np.ndarray,
    r0: float,
    horizon: float,
) -> Tuple[Dict[str, np.ndarray], Genealogy]:
    """Run Ogata thinning generation from an arbitrary direction w0."""
    max_events = data.cfg["simulate"]["horizon_events"]
    safety_factor = data.cfg["simulate"].get("safety_factor", 1.5)
    seed = {
        "W": np.array([w0], dtype=np.float32),
        "R": np.array([r0], dtype=np.float32),
        "dT": np.array([1.0], dtype=np.float32),
        "T": np.array([0.0], dtype=np.float32),
    }
    events, genealogy = generate_with_genealogy(
        seed, max_events, horizon, data.model, data.q_model, safety_factor,
    )
    return events, genealogy


# ── Subcriticality diagnostics ───────────────────────────────────────────


def compute_subcriticality_diagnostics(kernel: np.ndarray) -> Dict[str, Any]:
    """Compute subcriticality metrics from kernel matrix."""
    row_sums = kernel.sum(axis=-1)
    max_row_sum = float(row_sums.max()) if len(row_sums) > 0 else 0.0
    try:
        eigenvalues = np.linalg.eigvals(kernel)
        spectral_radius = float(np.max(np.abs(eigenvalues)))
    except Exception:
        spectral_radius = max_row_sum
    return {
        "row_sums": row_sums,
        "max_row_sum": max_row_sum,
        "spectral_radius": spectral_radius,
    }


# ── Coexistence ──────────────────────────────────────────────────────────


def compute_coexistence(
    T: np.ndarray,
    parents: np.ndarray,
    clusters: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute number of simultaneously active cascades over time.

    A cascade is "active" between its first and last event time.
    Returns (times, coexistence_count) aligned to event times T.
    """
    # Compute active intervals per cluster
    intervals = []
    for imm_idx, members in clusters.items():
        if len(members) == 0:
            continue
        t_start = T[members[0]]
        t_end = T[members[-1]]
        intervals.append((t_start, t_end))

    coex = np.zeros(len(T), dtype=np.int32)
    for i, t in enumerate(T):
        count = sum(1 for (ts, te) in intervals if ts <= t <= te)
        coex[i] = count

    return T, coex


# ── Galton-Watson termination bounds ─────────────────────────────────────


def cascade_termination_bounds(
    nu: float,
    max_n: int = 50,
) -> Dict[str, np.ndarray]:
    """Compute Galton-Watson cascade termination bounds.

    For a subcritical branching process with mean offspring nu < 1:
    - E[|C|] <= 1 / (1 - nu)
    - P(|C| >= n) <= nu^{n-1}

    Returns dict with ns, survival_prob, expected_size.
    """
    ns = np.arange(1, max_n + 1)
    survival_prob = np.power(nu, ns - 1)
    expected_size = 1.0 / (1.0 - nu) if nu < 1.0 else float("inf")
    return {
        "ns": ns,
        "survival_prob": survival_prob,
        "expected_size": expected_size,
    }


# ── Per-component loss diagnostics ────────────────────────────────────


def compute_loss_components(
    data: AppData,
    T: np.ndarray,
    R: np.ndarray,
    W: np.ndarray,
    dT: np.ndarray,
    u: np.ndarray | None = None,
    window: int = 128,
    tokens: np.ndarray | None = None,
    q_model_override=None,
) -> Optional[Dict[str, float]]:
    """Compute per-component mean log-likelihoods (log_p_w, log_p_r, log_p_t).

    Returns dict with mean values for each component, or None if too few events.
    """
    model = data.model
    if len(T) < 3:
        return None

    if u is None:
        q = q_model_override if q_model_override is not None else data.q_model
        with torch.no_grad():
            u = q(torch.tensor(W, dtype=torch.float32)).numpy()

    max_len = getattr(model.cfg, "max_len", window)
    window = min(window, max_len, len(T))
    start = max(0, len(T) - window)

    T_w = T[start:]
    R_w = R[start:]
    W_w = W[start:]
    dT_w = dT[start:]
    u_w = u[start:]

    d_input = model.input_proj.in_features
    if tokens is not None:
        tok = tokens[start:]
    else:
        u_thresh = None
        if d_input > 14:
            q = q_model_override if q_model_override is not None else data.q_model
            with torch.no_grad():
                u_thresh = q(torch.tensor(W_w, dtype=torch.float32)).numpy()
        tok = build_tokens_from_arrays(W_w, R_w, dT_w, u=u_thresh)

    if tok.shape[1] > d_input:
        tok = tok[:, :d_input]

    tokens_t = torch.tensor(tok[None, :, :], dtype=torch.float32)
    T_tensor = torch.tensor(T_w[None, :], dtype=torch.float32)
    R_tensor = torch.tensor(R_w[None, :], dtype=torch.float32)
    W_tensor = torch.tensor(W_w[None, :, :], dtype=torch.float32)
    dT_tensor = torch.tensor(dT_w[None, :], dtype=torch.float32)
    u_tensor = torch.tensor(u_w[None, :], dtype=torch.float32)

    with torch.no_grad():
        out = model.log_likelihood(
            tokens_t[:, :-1, :],
            W_tensor[:, 1:, :],
            R_tensor[:, 1:],
            dT_tensor[:, 1:],
            T_tensor[:, :-1],
            R_tensor[:, :-1],
            u_tensor[:, 1:],
            W_in=W_tensor[:, :-1, :],
        )

    return {
        "log_p_w": float(out["log_p_w"].mean().item()),
        "log_p_r": float(out["log_p_r"].mean().item()),
        "log_p_t": float(out["log_p_t"].mean().item()),
    }


# ── Statistical tests ────────────────────────────────────────────────


def compute_marginal_tests(
    real_vals: np.ndarray,
    gen_vals: np.ndarray,
) -> Dict[str, float]:
    """KS test + Wasserstein distance between two samples.

    Returns {"ks_stat", "ks_pval", "wasserstein"}.
    """
    ks_stat, ks_pval = ks_2samp(real_vals, gen_vals)
    w_dist = wasserstein_distance(real_vals, gen_vals)
    return {"ks_stat": float(ks_stat), "ks_pval": float(ks_pval), "wasserstein": float(w_dist)}


# ── Autocorrelation ──────────────────────────────────────────────────


def compute_autocorrelation(x: np.ndarray, max_lag: int = 30) -> np.ndarray:
    """Compute ACF of a 1-D array up to max_lag.

    Returns array of length max_lag+1 (lag 0 = 1.0).
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 3:
        return np.ones(min(max_lag + 1, n))
    x_centered = x - x.mean()
    var = float(np.dot(x_centered, x_centered))
    if var < 1e-15:
        return np.ones(max_lag + 1)
    acf = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag >= n:
            acf[lag] = 0.0
        else:
            acf[lag] = float(np.dot(x_centered[: n - lag], x_centered[lag:])) / var
    return acf


# ── Cluster sizes ────────────────────────────────────────────────────


def compute_cluster_sizes(genealogy: "Genealogy") -> np.ndarray:
    """Return array of cluster sizes from a Genealogy object."""
    return np.array([len(members) for members in genealogy.clusters.values()])


# ── Dominant asset frequency ─────────────────────────────────────────


def compute_dominant_freq(W: np.ndarray, d: int) -> np.ndarray:
    """Fraction of events dominated by each asset.

    Returns array of length d summing to 1.
    """
    dominant = np.argmax(np.abs(W), axis=1)
    counts = np.bincount(dominant, minlength=d)
    total = counts.sum()
    return counts.astype(np.float64) / max(total, 1)


# ── Transition matrix ────────────────────────────────────────────────


def compute_transition_matrix(W: np.ndarray, d: int) -> np.ndarray:
    """Row-stochastic transition matrix of dominant asset sequences.

    Returns (d, d) matrix.
    """
    dominant = np.argmax(np.abs(W), axis=1)
    counts = np.zeros((d, d), dtype=np.float64)
    for i in range(len(dominant) - 1):
        counts[dominant[i], dominant[i + 1]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return counts / row_sums


def transition_matrix_distance(P_real: np.ndarray, P_gen: np.ndarray) -> float:
    """Frobenius norm ||P_real - P_gen||_F."""
    return float(np.linalg.norm(P_real - P_gen))


# ── C2ST: Classifier Two-Sample Test ─────────────────────────────────


def compute_c2st(
    real_features: np.ndarray,
    gen_features: np.ndarray,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """Classifier Two-Sample Test via stratified k-fold logistic regression.

    Features per event: [R, log(dT), POC, W...] (6-dim for d=3).
    Returns {"auc_mean", "auc_std", "fpr", "tpr"}.
    AUC ~ 0.5 = indistinguishable (good), AUC ~ 1.0 = trivially separable (bad).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.preprocessing import StandardScaler

    n_real = len(real_features)
    n_gen = len(gen_features)
    X = np.vstack([real_features, gen_features])
    y = np.concatenate([np.zeros(n_real), np.ones(n_gen)])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    all_probs = np.zeros(len(y))

    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X_train, y[train_idx])
        probs = clf.predict_proba(X_test)[:, 1]
        all_probs[test_idx] = probs
        aucs.append(roc_auc_score(y[test_idx], probs))

    fpr, tpr, _ = roc_curve(y, all_probs)
    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "fpr": fpr,
        "tpr": tpr,
    }
