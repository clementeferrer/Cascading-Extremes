"""Utility re-exports from cascades.utils plus sphere helpers."""

import math

import numpy as np
import torch

from cascades.utils import EmpiricalCDF, ensure_dir, load_config, save_json, seed_all  # noqa: F401


def normalize_to_sphere(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize rows of *x* to the L2 unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def log_vmf_normalizing(kappa: torch.Tensor, d: int) -> torch.Tensor:
    """Log normalizing constant C_d(kappa) of the von Mises-Fisher distribution.

    For d=3:  log C_3(kappa) = log(kappa) - log(4*pi) - log(sinh(kappa))
    General:  log C_d(kappa) = (d/2-1)*log(kappa) - (d/2)*log(2*pi) - log(I_{d/2-1}(kappa))

    Uses the d=3 analytic form when applicable, otherwise a Bessel approximation.
    """
    eps = 1e-8
    kappa = torch.clamp(kappa, min=eps)

    if d == 3:
        # C_3(kappa) = kappa / (4*pi*sinh(kappa))
        return torch.log(kappa) - math.log(4.0 * math.pi) - torch.log(torch.sinh(kappa) + eps)

    # General case: use the relation I_{v}(kappa) ~ exp(kappa)/sqrt(2*pi*kappa) for large kappa
    # For moderate kappa use torch.special.i0e / i1e when v=0,1 or the saddle-point approx.
    v = d / 2.0 - 1.0
    # Saddle-point approximation: log I_v(kappa) ~ kappa - 0.5*log(2*pi*kappa) for large kappa
    # More accurate: log I_v(kappa) ~ kappa - 0.5*log(2*pi*kappa) - (4*v^2-1)/(8*kappa)
    log_iv_approx = kappa - 0.5 * math.log(2.0 * math.pi) - 0.5 * torch.log(kappa + eps)
    log_c = v * torch.log(kappa) - (d / 2.0) * math.log(2.0 * math.pi) - log_iv_approx
    return log_c
