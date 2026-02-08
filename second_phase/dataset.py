"""Data pipeline for full-sphere cascading extremes.

L2 radial-angular decomposition, spherical features, token building,
and PyTorch Dataset for training.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def compute_radial_angular_l2(X: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """L2 radial-angular decomposition: R = ||X||_2, W = X / ||X||_2 in S^{d-1}."""
    R = np.linalg.norm(X, axis=1)
    R = np.maximum(R, eps)
    W = X / R[:, None]
    return R, W


def spherical_features(W: np.ndarray) -> np.ndarray:
    """Compute spherical features xi(W) = [W, W^2, W_i*W_j for i<j].

    For d assets this gives d + d + d*(d-1)/2 = d*(d+3)/2 features.
    For d=3: 3 + 3 + 3 = 9 features.
    """
    d = W.shape[1]
    parts = [W, W ** 2]
    # Cross-products W_i * W_j for i < j
    for i in range(d):
        for j in range(i + 1, d):
            parts.append((W[:, i] * W[:, j])[:, None])
    return np.concatenate(parts, axis=1)


def build_token_sphere(w: np.ndarray, r: float, dt: float, eps: float = 1e-8) -> np.ndarray:
    """Build a single token: [W, xi(W), log R, log dT]."""
    xi = spherical_features(w[None, :])[0]
    return np.concatenate([w, xi, [np.log(r + eps), np.log(dt + eps)]], axis=0)


def build_events_sphere(
    X_t: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    u_tau_fn,
    eps: float = 1e-8,
) -> Dict[str, np.ndarray]:
    """Build event sequence from Laplace-margined data with spherical decomposition.

    Parameters
    ----------
    X_t : DataFrame with Laplace-margined values (can be negative).
    timestamps : DatetimeIndex aligned with X_t.
    u_tau_fn : callable W -> u_tau(W), directional threshold.

    Returns
    -------
    dict with keys: T, R, W, u, dT, tokens
    """
    X = X_t.values
    R, W = compute_radial_angular_l2(X, eps=eps)
    u = u_tau_fn(W)
    mask = R > u
    idx = np.where(mask)[0]

    if len(idx) < 2:
        raise ValueError("Not enough exceedances to build events. Try lowering tau.")

    times = timestamps.to_numpy(dtype="datetime64[ns]")[idx]
    t0 = times[0]
    T = (times - t0) / np.timedelta64(1, "h")
    T = T.astype(np.float32)

    R_e = R[idx].astype(np.float32)
    W_e = W[idx].astype(np.float32)
    u_e = u[idx].astype(np.float32)

    dT = np.diff(T, prepend=T[0])
    dT[0] = np.median(dT[1:]) if len(dT) > 1 else 1.0

    xi_e = spherical_features(W_e).astype(np.float32)

    tokens = np.concatenate(
        [
            W_e,
            xi_e,
            np.log(R_e + eps)[:, None],
            np.log(dT + eps)[:, None],
        ],
        axis=1,
    ).astype(np.float32)

    return {
        "T": T,
        "R": R_e,
        "W": W_e,
        "u": u_e,
        "dT": dT.astype(np.float32),
        "tokens": tokens,
    }


@dataclass
class SequenceConfig:
    seq_len: int = 64
    stride: int = 1


class SphereEventSequenceDataset(Dataset):
    """Sliding-window dataset over sphere event sequences."""

    def __init__(self, events: Dict[str, np.ndarray], cfg: SequenceConfig):
        self.tokens = events["tokens"]
        self.W = events["W"]
        self.R = events["R"]
        self.u = events["u"]
        self.dT = events["dT"]
        self.T = events["T"]
        self.seq_len = cfg.seq_len
        self.stride = cfg.stride

        self.indices = []
        max_start = len(self.tokens) - self.seq_len
        for start in range(0, max_start + 1, self.stride):
            self.indices.append(start)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = self.indices[idx]
        end = start + self.seq_len
        s = slice(start, end)
        return {
            "tokens": torch.tensor(self.tokens[s], dtype=torch.float32),
            "W": torch.tensor(self.W[s], dtype=torch.float32),
            "R": torch.tensor(self.R[s], dtype=torch.float32),
            "u": torch.tensor(self.u[s], dtype=torch.float32),
            "dT": torch.tensor(self.dT[s], dtype=torch.float32),
            "T": torch.tensor(self.T[s], dtype=torch.float32),
        }
