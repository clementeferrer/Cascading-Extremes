from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def compute_radial_angular(X: np.ndarray, eps: float = 1.0e-8) -> Tuple[np.ndarray, np.ndarray]:
    R = np.sum(X, axis=1)
    R = np.maximum(R, eps)
    W = X / R[:, None]
    W = np.clip(W, eps, None)
    W = W / W.sum(axis=1, keepdims=True)
    return R, W


def zeta(W: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    lw = np.log(W + eps)
    return lw - lw.mean(axis=1, keepdims=True)


def build_events(
    X_t: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    u_tau_fn,
    eps: float = 1.0e-8,
) -> Dict[str, np.ndarray]:
    X = X_t.values
    R, W = compute_radial_angular(X, eps=eps)
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

    zeta_e = zeta(W_e, eps=eps).astype(np.float32)

    tokens = np.concatenate(
        [
            W_e,
            zeta_e,
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


class EventSequenceDataset(Dataset):
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
        slice_ = slice(start, end)
        return {
            "tokens": torch.tensor(self.tokens[slice_], dtype=torch.float32),
            "W": torch.tensor(self.W[slice_], dtype=torch.float32),
            "R": torch.tensor(self.R[slice_], dtype=torch.float32),
            "u": torch.tensor(self.u[slice_], dtype=torch.float32),
            "dT": torch.tensor(self.dT[slice_], dtype=torch.float32),
            "T": torch.tensor(self.T[slice_], dtype=torch.float32),
        }
