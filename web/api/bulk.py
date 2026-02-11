"""Compute and cache bulk (all) observations for the 3D viewer.

Returns R*W positions for ALL observations, not just extremes.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT_DIR / "artifacts" / "phase2" / "bulk_observations.npz"


def _compute_bulk() -> np.ndarray:
    """Compute R*W for all observations from raw data."""
    from second_phase.preprocess import compute_log_returns, fit_garch, standardize_laplace
    from second_phase.dataset import compute_radial_angular_l2
    from cascades.data import download as download_data
    from cascades.utils import load_config

    cfg = load_config(str(ROOT_DIR / "configs" / "phase2.yaml"))
    data_cfg = cfg["data"]

    prices = download_data(
        data_cfg["symbols"],
        period=data_cfg.get("period", "730d"),
        interval=data_cfg.get("interval", "1h"),
        cache_dir=data_cfg.get("cache_dir", "data/raw"),
        price_field=data_cfg.get("price_field", "Close"),
    )

    returns = compute_log_returns(prices)
    residuals, _ = fit_garch(returns, dist=cfg["preprocess"].get("garch_dist", "t"))
    X, _ = standardize_laplace(residuals, pit_clip=cfg["preprocess"].get("pit_clip", 1e-6))
    X = X.dropna()

    R_all, W_all = compute_radial_angular_l2(X.values, eps=cfg["extremes"].get("eps", 1e-8))
    positions = W_all * R_all[:, None]  # (N, 3)

    # Cache to disk
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(CACHE_PATH), positions=positions)

    return positions


_cached_positions: np.ndarray | None = None


def get_bulk_positions() -> np.ndarray:
    """Return cached bulk positions (N, 3), computing if needed."""
    global _cached_positions
    if _cached_positions is not None:
        return _cached_positions

    if CACHE_PATH.exists():
        data = np.load(str(CACHE_PATH))
        _cached_positions = data["positions"]
        return _cached_positions

    _cached_positions = _compute_bulk()
    return _cached_positions
