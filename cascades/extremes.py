from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class QuantileModelConfig:
    hidden_sizes: Tuple[int, ...] = (64, 64)
    lr: float = 1.0e-3
    epochs: int = 200
    batch_size: int = 256


class DirectionalQuantileMLP(nn.Module):
    def __init__(self, d_in: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        layers = []
        last = d_in
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        out = self.net(w)
        return self.softplus(out).squeeze(-1)


def pinball_loss(y: torch.Tensor, y_hat: torch.Tensor, tau: float) -> torch.Tensor:
    diff = y - y_hat
    return torch.maximum(tau * diff, (tau - 1.0) * diff).mean()


def fit_directional_threshold(
    W: np.ndarray,
    R: np.ndarray,
    tau: float,
    model: str = "mlp",
    config: Optional[QuantileModelConfig] = None,
    device: str = "cpu",
) -> Tuple[Callable[[np.ndarray], np.ndarray], Dict[str, np.ndarray], DirectionalQuantileMLP]:
    if model != "mlp":
        raise ValueError("Only MLP quantile model is supported in this MVP")

    cfg = config or QuantileModelConfig()
    W_t = torch.tensor(W, dtype=torch.float32)
    R_t = torch.tensor(R, dtype=torch.float32)

    dataset = TensorDataset(W_t, R_t)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    net = DirectionalQuantileMLP(W.shape[1], cfg.hidden_sizes).to(device)
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
            out = net(w_t).cpu().numpy()
            return out

    meta = {
        "tau": np.array([tau], dtype=np.float32),
    }
    return u_tau, meta, net
