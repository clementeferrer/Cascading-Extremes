"""Data loading and caching for the Streamlit Cascade Explorer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import streamlit as st
import torch

from second_phase.dataset import spherical_features
from second_phase.extremes import SphericalQuantileMLP, QuantileModelConfig, ConstantThreshold
from second_phase.model import SphericalCascadeTransformer
from second_phase.simulate import load_model, load_quantile_model
from cascades.utils import load_config


@dataclass
class AppData:
    """Container for all loaded data, models, and config."""
    cfg: Dict[str, Any]
    symbols: list
    model: SphericalCascadeTransformer
    q_model: Any  # SphericalQuantileMLP or ConstantThreshold
    real_events: Dict[str, np.ndarray]
    sim_events: Optional[Dict[str, np.ndarray]]
    genealogy_data: Optional[Dict[str, np.ndarray]]
    bulk_positions: Optional[np.ndarray]
    cdfs_data: Optional[Dict[str, np.ndarray]]
    tau: float
    d_assets: int
    threshold_mode: str = "directional"


def _ensure_tokens(events: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Ensure events dict has 'tokens' and 'u' keys."""
    if "tokens" not in events or events["tokens"] is None:
        W, R, dT = events["W"], events["R"], events["dT"]
        xi = spherical_features(W).astype(np.float32)
        tokens = np.concatenate(
            [W, xi, np.log(R + 1e-8)[:, None], np.log(dT + 1e-8)[:, None]],
            axis=1,
        ).astype(np.float32)
        events["tokens"] = tokens
    if "u" not in events:
        events["u"] = np.ones_like(events["R"])
    return events


@st.cache_resource
def _load_model(model_path: str, _mtime: float = 0.0) -> SphericalCascadeTransformer:
    return load_model(model_path)


@st.cache_resource
def _load_qmodel(q_path: str, d_assets: int, hidden_sizes: tuple) -> SphericalQuantileMLP:
    cfg = QuantileModelConfig(hidden_sizes=hidden_sizes)
    return load_quantile_model(q_path, d_assets, cfg)


@st.cache_data
def _load_events(path: str) -> Dict[str, np.ndarray]:
    return _ensure_tokens(dict(np.load(path, allow_pickle=True)))


@st.cache_data
def _load_npz(path: str) -> Dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=True))


def load_all_data(
    cfg_path: str = "configs/phase2.yaml",
    artifact_dir: str | None = None,
    events_path: str | None = None,
) -> AppData:
    """Load all data, models, and config. Returns an AppData dataclass."""
    cfg = load_config(cfg_path)
    if artifact_dir is None:
        artifact_dir = cfg.get("artifact_dir", "artifacts/phase2")
    if events_path is None:
        processed_dir = cfg.get("processed_dir", "data/processed_phase2")
        events_path = str(Path(processed_dir) / "events.npz")

    art = Path(artifact_dir)

    # Symbols and threshold mode from meta.json
    symbols = cfg["data"]["symbols"]
    threshold_mode = cfg["extremes"].get("threshold_mode", "directional")
    meta_path = art / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
            symbols = meta.get("symbols", symbols)
            threshold_mode = meta.get("threshold_mode", threshold_mode)

    # Models
    model_file = art / "model.pt"
    model = _load_model(str(model_file), _mtime=model_file.stat().st_mtime)
    d_assets = model.d_assets

    if threshold_mode == "global":
        u_global = meta["u_global"]
        q_model = ConstantThreshold(u_global)
    else:
        q_model = _load_qmodel(
            str(art / "quantile_model.pt"),
            d_assets,
            tuple(cfg["extremes"]["quantile_model"]["hidden_sizes"]),
        )

    # Real events
    real_events = _load_events(events_path)

    # Simulated events
    sim_events = None
    sim_path = art / "simulated_events.npz"
    if sim_path.exists():
        sim_events = _ensure_tokens(dict(np.load(str(sim_path), allow_pickle=True)))

    # Genealogy
    genealogy_data = None
    gen_path = art / "genealogy.npz"
    if gen_path.exists():
        genealogy_data = _load_npz(str(gen_path))

    # Bulk observations
    bulk_positions = None
    bulk_path = art / "bulk_observations.npz"
    if bulk_path.exists():
        bulk_data = _load_npz(str(bulk_path))
        bulk_positions = bulk_data.get("positions")

    # CDFs
    cdfs_data = None
    cdfs_path = art / "cdfs.npz"
    if cdfs_path.exists():
        cdfs_data = _load_npz(str(cdfs_path))

    tau = cfg["extremes"]["tau"]

    return AppData(
        cfg=cfg,
        symbols=symbols,
        model=model,
        q_model=q_model,
        real_events=real_events,
        sim_events=sim_events,
        genealogy_data=genealogy_data,
        bulk_positions=bulk_positions,
        cdfs_data=cdfs_data,
        tau=tau,
        d_assets=d_assets,
        threshold_mode=threshold_mode,
    )


def load_global_data() -> AppData:
    """Load data trained with global (scalar) threshold."""
    return load_all_data(cfg_path="configs/phase2_global.yaml")


def load_var_data() -> AppData:
    """Load VAR(1) simulated data using the same ``load_all_data`` pipeline."""
    return load_all_data(cfg_path="configs/phase2_var.yaml")
