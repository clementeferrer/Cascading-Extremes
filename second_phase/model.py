"""SphericalCascadeTransformer — full-sphere model with vMF direction,
truncated Gamma magnitude, Hawkes + attenuation time head, and subcriticality penalty.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

from second_phase.utils import log_vmf_normalizing


def _sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding (Vaswani et al. 2017)."""
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


@dataclass
class ModelConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    vmf_components: int = 4
    hawkes_hidden: int = 64
    gauge_hidden: int = 64
    attenuation_hidden: int = 32
    max_len: int = 1024
    a_min: float = 1e-3
    tau_min: float = 1.0  # minimum decay timescale (hours)
    subcrit_margin: float = 0.95
    context_shape: bool = False
    shape_hidden: int = 64
    gauge_dt: bool = False


def make_mlp(in_dim: int, hidden: Tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers = []
    last = in_dim
    for h in hidden:
        layers.append(nn.Linear(last, h))
        layers.append(nn.ReLU())
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class SphericalCascadeTransformer(nn.Module):
    """Causal transformer for cascading extremes on the full sphere.

    Three heads:
    - Direction: mixture of von Mises-Fisher on S^{d-1}
    - Magnitude: truncated Gamma with gauge network
    - Time: Hawkes intensity with directional attenuation
    """

    def __init__(self, d_input: int, d_assets: int, cfg: ModelConfig):
        super().__init__()
        self.d_assets = d_assets
        self.cfg = cfg
        self.eps = 1e-8

        # Encoder
        self.input_proj = nn.Linear(d_input, cfg.d_model)
        self.pos_emb = nn.Parameter(_sinusoidal_encoding(cfg.max_len, cfg.d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=4 * cfg.d_model,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)

        # ---- Direction head: mixture of vMF ----
        K = cfg.vmf_components
        self.pi_head = nn.Linear(cfg.d_model, K)
        # Each component has a mean direction (d_assets) and concentration (1)
        self.mu_heads = nn.ModuleList([nn.Linear(cfg.d_model, d_assets) for _ in range(K)])
        self.kappa_head = nn.Linear(cfg.d_model, K)

        # ---- Magnitude head ----
        gauge_in_dim = cfg.d_model + 1 + d_assets + (1 if cfg.gauge_dt else 0)
        self.gauge_net = make_mlp(gauge_in_dim, (cfg.gauge_hidden, cfg.gauge_hidden), 1)
        if cfg.context_shape:
            self.shape_net = make_mlp(cfg.d_model + d_assets, (cfg.shape_hidden, cfg.shape_hidden), 1)
        else:
            self.a_param = nn.Parameter(torch.tensor(1.0))

        # ---- Time head: Hawkes with attenuation ----
        self.mu_net = make_mlp(cfg.d_model, (cfg.hawkes_hidden, cfg.hawkes_hidden), 1)
        self.k_net = make_mlp(2 * cfg.d_model, (cfg.hawkes_hidden, cfg.hawkes_hidden), 1)
        self.tau_net = make_mlp(2 * cfg.d_model, (cfg.hawkes_hidden, cfg.hawkes_hidden), 1)
        self.phi_net = make_mlp(1, (cfg.hawkes_hidden,), 1)

        # Attenuation: sigmoid(MLP([W_i, W_j])) in (0, 1]
        self.attenuation_net = make_mlp(2 * d_assets, (cfg.attenuation_hidden, cfg.attenuation_hidden), 1)

        self.softplus = nn.Softplus()

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = tokens.shape
        x = self.input_proj(tokens)
        pos = self.pos_emb[:seq_len].unsqueeze(0)
        x = x + pos
        mask = torch.triu(torch.ones(seq_len, seq_len, device=tokens.device), diagonal=1).bool()
        return self.encoder(x, mask=mask)

    # ------------------------------------------------------------------
    # Direction head (mixture of vMF)
    # ------------------------------------------------------------------

    def vmf_params(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (pi, mu, kappa) from hidden states.

        pi : (batch, seq, K) mixture weights
        mu : (batch, seq, K, d) mean directions (unit vectors)
        kappa : (batch, seq, K) concentrations (positive)
        """
        pi = torch.softmax(self.pi_head(h), dim=-1)

        mu_list = []
        for head in self.mu_heads:
            raw = head(h)  # (batch, seq, d)
            norm = torch.norm(raw, dim=-1, keepdim=True).clamp(min=self.eps)
            mu_list.append(raw / norm)
        mu = torch.stack(mu_list, dim=-2)  # (batch, seq, K, d)

        kappa = self.softplus(self.kappa_head(h)) + self.eps  # (batch, seq, K)
        return pi, mu, kappa

    # ------------------------------------------------------------------
    # Magnitude head
    # ------------------------------------------------------------------

    def gauge_rate(
        self,
        h: torch.Tensor,
        log_r: torch.Tensor,
        w_next: torch.Tensor,
        log_dt: torch.Tensor = None,
    ) -> torch.Tensor:
        parts = [h, log_r.unsqueeze(-1), w_next]
        if self.cfg.gauge_dt:
            if log_dt is not None:
                parts.append(log_dt.unsqueeze(-1))
            else:
                parts.append(torch.zeros_like(log_r.unsqueeze(-1)))
        x = torch.cat(parts, dim=-1)
        rate = self.softplus(self.gauge_net(x)) + self.eps
        return rate.squeeze(-1)

    # ------------------------------------------------------------------
    # Time head (Hawkes + attenuation)
    # ------------------------------------------------------------------

    def hawkes_intensity(
        self,
        h: torch.Tensor,
        T: torch.Tensor,
        log_r: torch.Tensor,
        W: torch.Tensor = None,
        return_kernel: bool = False,
    ):
        """Compute Hawkes intensity with optional directional attenuation.

        Parameters
        ----------
        h : (batch, seq, d_model)
        T : (batch, seq)
        log_r : (batch, seq)
        W : (batch, seq, d_assets) — if provided, compute attenuation kappa(W_i, W_j)

        Returns
        -------
        lam, psi [, kernel]
        """
        batch, seq_len, d_model = h.shape
        mu = self.softplus(self.mu_net(h)).squeeze(-1)

        # Pairwise hidden state features
        h_i = h[:, :, None, :].expand(-1, -1, seq_len, -1)
        h_j = h[:, None, :, :].expand(-1, seq_len, -1, -1)
        pair = torch.cat([h_i, h_j], dim=-1)
        pair_flat = pair.reshape(-1, 2 * d_model)

        k_scale = self.softplus(self.k_net(pair_flat)).view(batch, seq_len, seq_len)
        tau = self.softplus(self.tau_net(pair_flat)).view(batch, seq_len, seq_len) + self.cfg.tau_min

        delta = T[:, :, None] - T[:, None, :]
        delta = torch.clamp(delta, min=0.0)
        causal_mask = (T[:, :, None] - T[:, None, :]) >= 0

        kernel = k_scale * torch.exp(-delta / tau)

        # Directional attenuation
        if W is not None:
            W_i = W[:, :, None, :].expand(-1, -1, seq_len, -1)
            W_j = W[:, None, :, :].expand(-1, seq_len, -1, -1)
            w_pair = torch.cat([W_i, W_j], dim=-1)
            w_pair_flat = w_pair.reshape(-1, 2 * self.d_assets)
            attn = torch.sigmoid(self.attenuation_net(w_pair_flat)).view(batch, seq_len, seq_len)
            kernel = kernel * attn

        # Magnitude impact
        phi = self.softplus(self.phi_net(log_r.unsqueeze(-1))).squeeze(-1)
        kernel = kernel * phi[:, None, :]
        kernel = kernel * causal_mask

        psi = kernel.sum(dim=-1)
        lam = mu + psi

        if return_kernel:
            return lam, psi, kernel
        return lam, psi

    # ------------------------------------------------------------------
    # Subcriticality
    # ------------------------------------------------------------------

    def subcriticality_penalty(self, kernel: torch.Tensor) -> torch.Tensor:
        """Penalty if max row sum of kernel exceeds subcrit_margin.

        kernel : (batch, seq, seq) — the excitation kernel matrix.
        Returns scalar penalty.
        """
        row_sums = kernel.sum(dim=-1)  # (batch, seq)
        max_row_sum = row_sums.max()
        margin = self.cfg.subcrit_margin
        excess = torch.clamp(max_row_sum - margin, min=0.0)
        return excess ** 2

    # ------------------------------------------------------------------
    # Log-likelihood
    # ------------------------------------------------------------------

    def log_likelihood(
        self,
        tokens: torch.Tensor,
        W_next: torch.Tensor,
        R_next: torch.Tensor,
        dT_next: torch.Tensor,
        T_in: torch.Tensor,
        R_in: torch.Tensor,
        u_next: torch.Tensor,
        W_in: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        h = self.encode(tokens)

        # --- Direction (vMF mixture) ---
        pi, mu, kappa = self.vmf_params(h)
        d = self.d_assets
        # W_next: (batch, seq, d) -> expand for K components
        w_exp = W_next.unsqueeze(-2).expand_as(mu)  # (batch, seq, K, d)
        kappa_exp = kappa  # (batch, seq, K)
        log_c = log_vmf_normalizing(kappa_exp, d)
        dot = (w_exp * mu).sum(dim=-1)  # (batch, seq, K)
        log_vmf = log_c + kappa_exp * dot  # (batch, seq, K)
        log_mix = torch.log(pi + self.eps) + log_vmf
        log_p_w = torch.logsumexp(log_mix, dim=-1)

        # --- Magnitude (truncated Gamma) ---
        log_r_in = torch.log(R_in + self.eps)
        log_dt_in = torch.log(dT_next + self.eps) if self.cfg.gauge_dt else None
        rate = self.gauge_rate(h, log_r_in, W_next, log_dt=log_dt_in)
        if self.cfg.context_shape:
            shape = self.softplus(self.shape_net(torch.cat([h, W_next], dim=-1))).squeeze(-1) + self.cfg.a_min
        else:
            shape = self.softplus(self.a_param) + self.cfg.a_min
        log_pdf = (
            shape * torch.log(rate)
            - torch.lgamma(shape)
            + (shape - 1.0) * torch.log(R_next + self.eps)
            - rate * R_next
        )
        tail = 1.0 - torch.special.gammainc(shape.detach(), rate * u_next)
        tail = torch.clamp(tail, min=1e-4)
        log_p_r = log_pdf - torch.log(tail)

        # --- Time (Hawkes + attenuation) ---
        lam, psi, kernel = self.hawkes_intensity(h, T_in, log_r_in, W=W_in, return_kernel=True)
        lam = torch.clamp(lam, min=self.eps)
        log_p_t = torch.log(lam) - lam * torch.clamp(dT_next, max=100.0)

        # --- Subcriticality penalty ---
        subcrit = self.subcriticality_penalty(kernel)

        return {
            "log_p_w": log_p_w,
            "log_p_r": log_p_r,
            "log_p_t": log_p_t,
            "lambda": lam,
            "psi": psi,
            "subcrit_penalty": subcrit,
            "kernel": kernel,
        }
