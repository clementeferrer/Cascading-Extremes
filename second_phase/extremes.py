"""Directional quantile threshold on the sphere.

Input features are [W, xi(W)] where W lives on S^{d-1} (full sphere).
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from second_phase.dataset import spherical_features


@dataclass
class QuantileModelConfig:
    hidden_sizes: Tuple[int, ...] = (64, 64)
    lr: float = 1e-3
    epochs: int = 200
    batch_size: int = 256


class SphericalQuantileMLP(nn.Module):
    """Directional quantile model on the sphere.

    Input: [W, xi(W)] of dimension d*(d+3)/2.
    Output: u_tau(w) > 0 via Softplus.
    """

    def __init__(self, d_assets: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        # forward concatenates [w, w^2, cross_products] = d + d + d*(d-1)/2 = d*(d+3)/2
        d_features = d_assets * (d_assets + 3) // 2
        layers = []
        last = d_features
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """w: (batch, d_assets) on the sphere."""
        xi = self._spherical_features_torch(w)
        x = torch.cat([w, xi], dim=-1)
        return self.softplus(self.net(x)).squeeze(-1)

    @staticmethod
    def _spherical_features_torch(w: torch.Tensor) -> torch.Tensor:
        d = w.shape[-1]
        parts = [w ** 2]
        for i in range(d):
            for j in range(i + 1, d):
                parts.append((w[..., i] * w[..., j]).unsqueeze(-1))
        return torch.cat(parts, dim=-1)


class ConstantThreshold:
    """Drop-in replacement for SphericalQuantileMLP that returns a fixed scalar.

    Matches the callable interface: __call__(W_tensor) -> u_tensor.
    """

    def __init__(self, u: float):
        self.u = u

    def __call__(self, W):
        if isinstance(W, np.ndarray):
            return np.full(W.shape[0], self.u, dtype=W.dtype)
        return torch.full((W.shape[0],), self.u, dtype=W.dtype)


def compute_global_threshold(R: np.ndarray, tau: float) -> float:
    """Simple global quantile threshold: u = quantile(R, tau)."""
    return float(np.quantile(R, tau))


def pinball_loss(y: torch.Tensor, y_hat: torch.Tensor, tau: float) -> torch.Tensor:
    diff = y - y_hat
    return torch.maximum(tau * diff, (tau - 1.0) * diff).mean()


def fit_sphere_threshold(
    W: np.ndarray,
    R: np.ndarray,
    tau: float,
    config: Optional[QuantileModelConfig] = None,
    device: str = "cpu",
) -> Tuple[Callable[[np.ndarray], np.ndarray], Dict[str, np.ndarray], SphericalQuantileMLP]:
    """Fit directional quantile on the sphere.

    Returns (u_tau_fn, meta, net).
    """
    cfg = config or QuantileModelConfig()
    d_assets = W.shape[1]

    W_t = torch.tensor(W, dtype=torch.float32)
    R_t = torch.tensor(R, dtype=torch.float32)

    dataset = TensorDataset(W_t, R_t)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    net = SphericalQuantileMLP(d_assets, cfg.hidden_sizes).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    net.train()
    for _ in range(cfg.epochs):
        for batch_w, batch_r in loader:
            batch_w = batch_w.to(device)
            batch_r = batch_r.to(device)
            pred = net(batch_w)
            loss = pinball_loss(batch_r, pred, tau)
            optim.zero_grad()
            loss.backward()
            optim.step()

    net.eval()

    def u_tau(w: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            w_t = torch.tensor(w, dtype=torch.float32, device=device)
            return net(w_t).cpu().numpy()

    meta = {"tau": np.array([tau], dtype=np.float32)}
    return u_tau, meta, net
