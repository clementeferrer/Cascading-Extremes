"""Ogata thinning simulation + prompt-based generation for Phase 2.

Implements exact point process simulation via Ogata's modified thinning algorithm
with parent sampling for genealogy tracking.
"""

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from second_phase.dataset import build_token_sphere, spherical_features
from second_phase.distributions import sample_trunc_gamma, sample_vmf_mixture
from second_phase.extremes import SphericalQuantileMLP, QuantileModelConfig
from second_phase.genealogy import build_genealogy, Genealogy
from second_phase.model import SphericalCascadeTransformer, ModelConfig
from second_phase.utils import load_config, ensure_dir


def load_model(path: str) -> SphericalCascadeTransformer:
    payload = torch.load(path, map_location="cpu")
    cfg = ModelConfig(**payload["model_cfg"])
    model = SphericalCascadeTransformer(d_input=payload["d_input"], d_assets=payload["d_assets"], cfg=cfg)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def load_quantile_model(path: str, d_assets: int, cfg: QuantileModelConfig) -> SphericalQuantileMLP:
    model = SphericalQuantileMLP(d_assets, cfg.hidden_sizes)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def _encode_history(model, W_list, R_list, dT_list, T_list):
    """Encode the current event history into transformer hidden states."""
    max_len = model.cfg.max_len
    start = max(0, len(W_list) - max_len)

    W_hist = np.array(W_list[start:], dtype=np.float32)
    R_hist = np.array(R_list[start:], dtype=np.float32)
    dT_hist = np.array(dT_list[start:], dtype=np.float32)
    T_hist = np.array(T_list[start:], dtype=np.float32)

    tokens = np.stack(
        [build_token_sphere(W_hist[i], R_hist[i], dT_hist[i]) for i in range(len(W_hist))],
        axis=0,
    )
    tokens_t = torch.tensor(tokens[None, :, :], dtype=torch.float32)

    with torch.no_grad():
        h = model.encode(tokens_t)
    return h, tokens_t, W_hist, R_hist, T_hist


def _compute_intensity(model, h, T_hist, R_hist, W_hist):
    """Get current Hawkes intensity from encoded history."""
    T_tensor = torch.tensor(T_hist[None, :], dtype=torch.float32)
    R_tensor = torch.tensor(R_hist[None, :], dtype=torch.float32)
    W_tensor = torch.tensor(W_hist[None, :, :], dtype=torch.float32)
    log_r = torch.log(R_tensor + 1e-8)
    with torch.no_grad():
        lam, psi = model.hawkes_intensity(h, T_tensor, log_r, W=W_tensor)
    return lam[0, -1].item(), psi[0, -1].item()


def _sample_candidate(model, h, R_hist, q_model, d_assets):
    """Sample a candidate mark (W, R) from the model's conditional distributions."""
    with torch.no_grad():
        pi, mu, kappa = model.vmf_params(h)
        pi_last = pi[0, -1]
        mu_last = mu[0, -1]
        kappa_last = kappa[0, -1]
        w_next = sample_vmf_mixture(pi_last, mu_last, kappa_last, d_assets).numpy()

        log_r_in = torch.tensor([[np.log(R_hist[-1] + 1e-8)]], dtype=torch.float32)
        h_last = h[:, -1:, :]
        w_t = torch.tensor(w_next[None, None, :], dtype=torch.float32)
        rate = model.gauge_rate(h_last, log_r_in, w_t).item()
        u_next = q_model(torch.tensor(w_next[None, :], dtype=torch.float32)).item()
        shape = model.softplus(model.a_param).item() + model.cfg.a_min
        r_next = sample_trunc_gamma(shape, rate, u_next)

    return w_next, r_next


def ogata_thinning(
    seed: Dict[str, np.ndarray],
    max_events: int,
    max_time: Optional[float],
    model: SphericalCascadeTransformer,
    q_model: SphericalQuantileMLP,
    safety_factor: float = 1.5,
) -> Dict[str, np.ndarray]:
    """Generate events using Ogata's thinning algorithm.

    Algorithm:
    1. Compute intensity bound Lambda* = safety_factor * lambda(T_last)
    2. Propose dt ~ Exp(Lambda*)
    3. Sample candidate mark (W, R) from vMF mixture + truncGamma
    4. Accept with prob lambda(t*, m*) / Lambda*
    5. On acceptance: append to history
    6. Stop when T > horizon or max_events reached

    Returns dict with T, dT, W, R, accepted (bool mask for Ogata diagnostics).
    """
    d_assets = model.d_assets

    W = [w for w in seed["W"]]
    R = list(seed["R"])
    dT = list(seed["dT"])
    T = list(seed["T"])
    accepted_mask = [True] * len(T)  # seed events are always "accepted"

    t0 = T[-1] if T else 0.0

    for _ in range(max_events):
        h, tokens_t, W_hist, R_hist, T_hist = _encode_history(model, W, R, dT, T)
        lam_current, _ = _compute_intensity(model, h, T_hist, R_hist, W_hist)

        # Upper bound for thinning
        lam_star = max(safety_factor * lam_current, 1e-4)

        # Propose inter-arrival time
        dt_proposed = np.random.exponential(1.0 / lam_star)
        t_proposed = T[-1] + dt_proposed

        if max_time is not None and (t_proposed - t0) > max_time:
            break

        # Sample candidate mark
        w_candidate, r_candidate = _sample_candidate(model, h, R_hist, q_model, d_assets)

        # Temporarily add candidate to compute intensity at proposed time
        W_temp = W + [w_candidate]
        R_temp = R + [r_candidate]
        dT_temp = dT + [dt_proposed]
        T_temp = T + [t_proposed]

        h_temp, _, W_hist_temp, R_hist_temp, T_hist_temp = _encode_history(
            model, W_temp, R_temp, dT_temp, T_temp
        )
        lam_at_candidate, _ = _compute_intensity(model, h_temp, T_hist_temp, R_hist_temp, W_hist_temp)

        # Accept/reject
        accept_prob = min(lam_at_candidate / lam_star, 1.0)
        if np.random.rand() < accept_prob:
            W.append(w_candidate)
            R.append(r_candidate)
            dT.append(dt_proposed)
            T.append(t_proposed)
            accepted_mask.append(True)
        else:
            accepted_mask.append(False)

    return {
        "T": np.array(T, dtype=np.float32),
        "dT": np.array(dT, dtype=np.float32),
        "W": np.array(W, dtype=np.float32),
        "R": np.array(R, dtype=np.float32),
        "accepted": np.array(accepted_mask[:len(T)], dtype=bool),
    }


def generate_with_genealogy(
    seed: Dict[str, np.ndarray],
    max_events: int,
    max_time: Optional[float],
    model: SphericalCascadeTransformer,
    q_model: SphericalQuantileMLP,
    safety_factor: float = 1.5,
) -> tuple:
    """Generate events via Ogata thinning and then compute genealogy.

    Returns (events_dict, Genealogy).
    """
    events = ogata_thinning(seed, max_events, max_time, model, q_model, safety_factor)

    # Compute full intensity decomposition for genealogy
    tokens = np.stack(
        [build_token_sphere(events["W"][i], events["R"][i], events["dT"][i])
         for i in range(len(events["W"]))],
        axis=0,
    )
    tokens_t = torch.tensor(tokens[None, :, :], dtype=torch.float32)

    with torch.no_grad():
        h = model.encode(tokens_t)
        T_tensor = torch.tensor(events["T"][None, :], dtype=torch.float32)
        R_tensor = torch.tensor(events["R"][None, :], dtype=torch.float32)
        W_tensor = torch.tensor(events["W"][None, :, :], dtype=torch.float32)
        log_r = torch.log(R_tensor + 1e-8)
        lam, psi, kernel = model.hawkes_intensity(h, T_tensor, log_r, W=W_tensor, return_kernel=True)

    lam_np = lam.squeeze(0).numpy()
    psi_np = psi.squeeze(0).numpy()
    kernel_np = kernel.squeeze(0).numpy()

    genealogy = build_genealogy(lam_np, psi_np, kernel_np)
    return events, genealogy


def prompt_generate(
    asset_idx: int,
    magnitude: float,
    horizon: float,
    max_events: int,
    model: SphericalCascadeTransformer,
    q_model: SphericalQuantileMLP,
    safety_factor: float = 1.5,
) -> tuple:
    """Generate a cascade starting from a single prompt event.

    Parameters
    ----------
    asset_idx : which asset triggers the initial shock (0-indexed)
    magnitude : magnitude of the initial shock (signed: positive=rally, negative=crash)
    horizon : max time horizon (hours)
    max_events : max events to generate
    model, q_model : trained models

    Returns (events_dict, Genealogy)
    """
    d = model.d_assets

    # Construct initial direction: unit vector along asset_idx with sign
    w0 = np.zeros(d, dtype=np.float32)
    w0[asset_idx] = np.sign(magnitude) if magnitude != 0 else 1.0
    # Normalize (it's already unit but be explicit)
    w0 = w0 / np.linalg.norm(w0)

    r0 = float(abs(magnitude))
    u0 = q_model(torch.tensor(w0[None, :], dtype=torch.float32)).item()
    if r0 < u0:
        r0 = u0 + 0.1  # ensure it's an exceedance

    seed = {
        "W": np.array([w0], dtype=np.float32),
        "R": np.array([r0], dtype=np.float32),
        "dT": np.array([1.0], dtype=np.float32),
        "T": np.array([0.0], dtype=np.float32),
    }

    return generate_with_genealogy(seed, max_events, horizon, model, q_model, safety_factor)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate Phase 2 cascades via Ogata thinning.")
    parser.add_argument("--config", default="configs/phase2.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifact_dir = cfg.get("artifact_dir", "artifacts/phase2")

    model = load_model(str(Path(artifact_dir) / "model.pt"))
    q_cfg = QuantileModelConfig(**cfg["extremes"]["quantile_model"])
    q_model = load_quantile_model(
        str(Path(artifact_dir) / "quantile_model.pt"),
        model.d_assets,
        q_cfg,
    )

    events_path = Path("data/processed_phase2/events.npz")
    events = np.load(events_path)
    seed_len = cfg["simulate"]["seed_len"]
    seed = {k: events[k][-seed_len:] for k in ["W", "R", "dT", "T"]}

    sim_events, genealogy = generate_with_genealogy(
        seed,
        max_events=cfg["simulate"]["horizon_events"],
        max_time=cfg["simulate"].get("max_time"),
        model=model,
        q_model=q_model,
        safety_factor=cfg["simulate"].get("safety_factor", 1.5),
    )

    ensure_dir(artifact_dir)
    np.savez(str(Path(artifact_dir) / "simulated_events.npz"), **sim_events)
    np.savez(
        str(Path(artifact_dir) / "genealogy.npz"),
        parents=genealogy.parents,
        immigrant_mask=genealogy.immigrant_mask,
        cascade_probs=genealogy.cascade_probs,
    )
    print(f"Simulation complete -> {artifact_dir}/simulated_events.npz")
    print(f"Genealogy saved -> {artifact_dir}/genealogy.npz")
    print(f"  Immigrants: {genealogy.immigrant_mask.sum()} / {len(genealogy.parents)}")
    print(f"  Clusters: {len(genealogy.clusters)}")


if __name__ == "__main__":
    main()
