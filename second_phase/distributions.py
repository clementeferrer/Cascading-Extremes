"""Core distributions for the full-sphere model.

- von Mises-Fisher (vMF) density and sampling (Wood's algorithm)
- Mixture of vMF sampling
- Truncated Gamma sampling
- Excitation kernel math
"""

import math
from typing import Tuple

import torch
from torch import Tensor

from second_phase.utils import log_vmf_normalizing


# ---------------------------------------------------------------------------
# von Mises-Fisher
# ---------------------------------------------------------------------------

def log_vmf_density(w: Tensor, mu: Tensor, kappa: Tensor, d: int) -> Tensor:
    """Log-density of vMF(mu, kappa) evaluated at w.

    Parameters
    ----------
    w : (..., d) unit vectors
    mu : (..., d) mean directions (unit vectors)
    kappa : (...) concentration parameters
    d : ambient dimension

    Returns
    -------
    log p(w | mu, kappa) : (...)
    """
    log_c = log_vmf_normalizing(kappa, d)
    dot = (w * mu).sum(dim=-1)
    return log_c + kappa * dot


def _sample_w_wood(kappa: float, d: int, n: int = 1) -> Tensor:
    """Sample the 'w' component in Wood's algorithm for vMF.

    Rejection sampling on [-1, 1] using the envelope from Wood (1994).
    Returns samples of cos(theta) where theta is the angle from the mean.
    """
    m = d - 1  # dimension of the tangent space
    b = (-2.0 * kappa + math.sqrt(4.0 * kappa ** 2 + m ** 2)) / m
    a = (m + 2.0 * kappa + math.sqrt(4.0 * kappa ** 2 + m ** 2)) / 4.0
    dd = (4.0 * a * b) / (1.0 + b) - m * math.log(m)

    results = []
    while len(results) < n:
        eps = torch.distributions.Beta(m / 2.0, m / 2.0).sample()
        w_candidate = (1.0 - (1.0 + b) * eps) / (1.0 - (1.0 - b) * eps)
        t = (2.0 * a * b) / (1.0 - (1.0 - b) * eps)
        u = torch.rand(1).item()
        accept = m * math.log(t) - t + dd >= math.log(u)
        if accept:
            results.append(w_candidate.item())

    return torch.tensor(results)


def _uniform_tangent(mu: Tensor) -> Tensor:
    """Sample a uniformly random unit vector orthogonal to mu."""
    d = mu.shape[-1]
    v = torch.randn(d)
    # Gram-Schmidt: remove component along mu
    v = v - (v @ mu) * mu
    norm = torch.norm(v)
    if norm < 1e-10:
        # Degenerate case: pick another random vector
        v = torch.randn(d)
        v = v - (v @ mu) * mu
        norm = torch.norm(v)
    return v / norm


def sample_vmf(mu: Tensor, kappa: float, d: int) -> Tensor:
    """Sample one vector from vMF(mu, kappa) on S^{d-1} using Wood's algorithm.

    Parameters
    ----------
    mu : (d,) mean direction (unit vector)
    kappa : concentration (scalar)
    d : ambient dimension

    Returns
    -------
    w : (d,) sampled unit vector
    """
    if kappa < 1e-6:
        # Nearly uniform: sample uniformly on sphere
        w = torch.randn(d)
        return w / torch.norm(w)

    # Step 1: sample cos(theta) via rejection
    cos_theta = _sample_w_wood(kappa, d, n=1)[0]
    sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta ** 2, min=0.0))

    # Step 2: sample tangent direction uniformly
    v = _uniform_tangent(mu)

    # Step 3: combine
    w = cos_theta * mu + sin_theta * v
    # Renormalize for safety
    return w / torch.norm(w)


def sample_vmf_mixture(pi: Tensor, mu: Tensor, kappa: Tensor, d: int) -> Tensor:
    """Sample from a mixture of vMF distributions.

    Parameters
    ----------
    pi : (K,) mixture weights (sum to 1)
    mu : (K, d) mean directions
    kappa : (K,) concentrations

    Returns
    -------
    w : (d,) sampled unit vector
    """
    k = torch.distributions.Categorical(pi).sample().item()
    return sample_vmf(mu[k], kappa[k].item(), d)


# ---------------------------------------------------------------------------
# Truncated Gamma
# ---------------------------------------------------------------------------

def sample_trunc_gamma(shape: float, rate: float, u: float, max_tries: int = 512) -> float:
    """Sample R ~ Gamma(shape, rate) truncated to R > u via rejection."""
    dist = torch.distributions.Gamma(torch.tensor(shape), torch.tensor(rate))
    last = u
    for _ in range(max_tries):
        r = dist.sample().item()
        last = r
        if r > u:
            return r
    return max(last, u + 1e-6)


# ---------------------------------------------------------------------------
# Excitation kernel components
# ---------------------------------------------------------------------------

def exponential_kernel(delta: Tensor, amplitude: Tensor, decay: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute A * exp(-delta / tau) for the excitation kernel.

    Parameters
    ----------
    delta : (...) non-negative time differences
    amplitude : (...) kernel amplitude A (positive)
    decay : (...) decay time-scale tau (positive)

    Returns
    -------
    kernel values (...)
    """
    return amplitude * torch.exp(-delta / (decay + eps))


def magnitude_impact(log_r: Tensor, phi_net: torch.nn.Module) -> Tensor:
    """Compute phi(R) = softplus(phi_net(log R))."""
    return torch.nn.functional.softplus(phi_net(log_r.unsqueeze(-1))).squeeze(-1)
