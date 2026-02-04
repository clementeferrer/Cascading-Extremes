from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn


@dataclass
class ModelConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    dirichlet_components: int = 4
    hawkes_hidden: int = 64
    gauge_hidden: int = 64
    max_len: int = 256
    a_min: float = 1.0e-3


def make_mlp(in_dim: int, hidden: Tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers = []
    last = in_dim
    for h in hidden:
        layers.append(nn.Linear(last, h))
        layers.append(nn.ReLU())
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class CascadingTransformer(nn.Module):
    def __init__(self, d_input: int, d_assets: int, cfg: ModelConfig):
        super().__init__()
        self.d_assets = d_assets
        self.cfg = cfg
        self.eps = 1.0e-8

        self.input_proj = nn.Linear(d_input, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(cfg.max_len, cfg.d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=4 * cfg.d_model,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)

        self.pi_head = nn.Linear(cfg.d_model, cfg.dirichlet_components)
        self.alpha_head = nn.Linear(cfg.d_model, cfg.dirichlet_components * d_assets)

        self.gauge_net = make_mlp(cfg.d_model + 1 + d_assets, (cfg.gauge_hidden, cfg.gauge_hidden), 1)
        self.mu_net = make_mlp(cfg.d_model, (cfg.hawkes_hidden, cfg.hawkes_hidden), 1)
        self.k_net = make_mlp(2 * cfg.d_model, (cfg.hawkes_hidden, cfg.hawkes_hidden), 1)
        self.tau_net = make_mlp(2 * cfg.d_model, (cfg.hawkes_hidden, cfg.hawkes_hidden), 1)
        self.phi_net = make_mlp(1, (cfg.hawkes_hidden,), 1)

        self.softplus = nn.Softplus()
        self.a_param = nn.Parameter(torch.tensor(1.0))

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = tokens.shape
        x = self.input_proj(tokens)
        pos = self.pos_emb[:seq_len].unsqueeze(0)
        x = x + pos
        mask = torch.triu(torch.ones(seq_len, seq_len, device=tokens.device), diagonal=1).bool()
        h = self.encoder(x, mask=mask)
        return h

    def dirichlet_params(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pi_logits = self.pi_head(h)
        pi = torch.softmax(pi_logits, dim=-1)
        alpha_raw = self.alpha_head(h)
        alpha = self.softplus(alpha_raw).view(h.shape[0], h.shape[1], self.cfg.dirichlet_components, self.d_assets)
        alpha = alpha + self.eps
        return pi, alpha

    def gauge_rate(self, h: torch.Tensor, log_r: torch.Tensor, w_next: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, log_r.unsqueeze(-1), w_next], dim=-1)
        rate = self.softplus(self.gauge_net(x)) + self.eps
        return rate.squeeze(-1)

    def hawkes_intensity(
        self,
        h: torch.Tensor,
        T: torch.Tensor,
        log_r: torch.Tensor,
        return_kernel: bool = False,
    ):
        batch, seq_len, d_model = h.shape
        mu = self.softplus(self.mu_net(h)).squeeze(-1)

        h_i = h[:, :, None, :]
        h_j = h[:, None, :, :]
        pair = torch.cat([h_i.expand(-1, -1, seq_len, -1), h_j.expand(-1, seq_len, -1, -1)], dim=-1)
        pair_flat = pair.reshape(-1, 2 * d_model)

        k_scale = self.softplus(self.k_net(pair_flat)).view(batch, seq_len, seq_len)
        tau = self.softplus(self.tau_net(pair_flat)).view(batch, seq_len, seq_len) + self.eps

        delta = T[:, :, None] - T[:, None, :]
        delta = torch.clamp(delta, min=0.0)
        mask = (T[:, :, None] - T[:, None, :]) >= 0
        kernel = k_scale * torch.exp(-delta / tau)

        phi = self.softplus(self.phi_net(log_r.unsqueeze(-1))).squeeze(-1)
        kernel = kernel * phi[:, None, :]
        kernel = kernel * mask

        psi = kernel.sum(dim=-1)
        lam = mu + psi
        if return_kernel:
            return lam, psi, kernel
        return lam, psi

    def log_likelihood(
        self,
        tokens: torch.Tensor,
        W_next: torch.Tensor,
        R_next: torch.Tensor,
        dT_next: torch.Tensor,
        T_in: torch.Tensor,
        R_in: torch.Tensor,
        u_next: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        h = self.encode(tokens)

        pi, alpha = self.dirichlet_params(h)
        dirichlet = torch.distributions.Dirichlet(alpha)
        log_w = dirichlet.log_prob(W_next.unsqueeze(-2).expand_as(alpha))
        log_mix = torch.log(pi + self.eps) + log_w
        log_p_w = torch.logsumexp(log_mix, dim=-1)

        log_r_in = torch.log(R_in + self.eps)
        rate = self.gauge_rate(h, log_r_in, W_next)
        shape = self.softplus(self.a_param) + self.cfg.a_min
        log_pdf = shape * torch.log(rate) - torch.lgamma(shape) + (shape - 1.0) * torch.log(R_next + self.eps) - rate * R_next
        # Detach tail term to avoid igamma autograd limitations in PyTorch.
        tail = 1.0 - torch.special.gammainc(shape.detach(), (rate.detach() * u_next.detach()))
        tail = torch.clamp(tail, min=self.eps)
        log_p_r = log_pdf - torch.log(tail)

        lam, psi = self.hawkes_intensity(h, T_in, log_r_in)
        lam = torch.clamp(lam, min=self.eps)
        log_p_t = torch.log(lam) - lam * dT_next

        return {
            "log_p_w": log_p_w,
            "log_p_r": log_p_r,
            "log_p_t": log_p_t,
            "lambda": lam,
            "psi": psi,
        }
