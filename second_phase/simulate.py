"""Ogata thinning simulation + prompt-based generation for Phase 2.

Implements exact point process simulation via Ogata's modified thinning algorithm
with parent sampling for genealogy tracking.
"""

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from second_phase.dataset import build_token_sphere, build_tokens_from_arrays, spherical_features
from second_phase.distributions import sample_trunc_gamma, sample_vmf_mixture
from second_phase.extremes import SphericalQuantileMLP, QuantileModelConfig, ConstantThreshold
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


def _encode_history(model, W_list, R_list, dT_list, T_list, q_model=None):
    """Encode the current event history into transformer hidden states."""
    max_len = model.cfg.max_len
    start = max(0, len(W_list) - max_len)

    W_hist = np.array(W_list[start:], dtype=np.float32)
    R_hist = np.array(R_list[start:], dtype=np.float32)
    dT_hist = np.array(dT_list[start:], dtype=np.float32)
    T_hist = np.array(T_list[start:], dtype=np.float32)

    # Check model's expected input dimension for backward compat
    d_input = model.input_proj.in_features

    # Compute thresholds for exceedance feature if q_model available and model expects enriched tokens
    u_hist = None
    if q_model is not None and d_input > 14:
        with torch.no_grad():
            u_hist = q_model(torch.tensor(W_hist, dtype=torch.float32)).numpy()

    tokens = build_tokens_from_arrays(W_hist, R_hist, dT_hist, u=u_hist)

    # Truncate to model's expected d_input (backward compat with 14-dim models)
    if tokens.shape[1] > d_input:
        tokens = tokens[:, :d_input]

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


def _clamp_temperature(temperature: float) -> float:
    """Clamp sampling temperature to a safe range."""
    return float(np.clip(temperature, 0.2, 3.0))


def _sample_candidate(model, h, R_hist, q_model, d_assets, temperature: float = 1.0, last_dt: float = 1.0):
    """Sample a candidate mark (W, R) from the model's conditional distributions."""
    temp = _clamp_temperature(temperature)
    with torch.no_grad():
        pi, mu, kappa = model.vmf_params(h)
        pi_last = pi[0, -1]
        mu_last = mu[0, -1]
        kappa_last = kappa[0, -1]
        # Temperature decoding: soften mixture logits and directional concentration.
        pi_last = torch.softmax(torch.log(pi_last + 1.0e-8) / temp, dim=-1)
        kappa_last = torch.clamp(kappa_last / temp, min=1.0e-6)
        w_next = sample_vmf_mixture(pi_last, mu_last, kappa_last, d_assets).numpy()

        log_r_in = torch.tensor([[np.log(R_hist[-1] + 1e-8)]], dtype=torch.float32)
        h_last = h[:, -1:, :]
        w_t = torch.tensor(w_next[None, None, :], dtype=torch.float32)
        log_dt_in = torch.tensor([[np.log(last_dt + 1e-8)]], dtype=torch.float32) if model.cfg.gauge_dt else None
        rate = model.gauge_rate(h_last, log_r_in, w_t, log_dt=log_dt_in).item()
        u_next = q_model(torch.tensor(w_next[None, :], dtype=torch.float32)).item()
        if model.cfg.context_shape:
            shape = model.softplus(model.shape_net(torch.cat([h_last, w_t], dim=-1))).item() + model.cfg.a_min
        else:
            shape = model.softplus(model.a_param).item() + model.cfg.a_min
        r_next = sample_trunc_gamma(shape, rate, u_next)

    return w_next, r_next


def autoregressive_generate(
    w0: np.ndarray,
    r0: float,
    max_time: float,
    model: SphericalCascadeTransformer,
    q_model: SphericalQuantileMLP,
    max_events_cap: int = 2000,
    temperature: float = 1.0,
    prompt: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """LLM-style autoregressive generation of cascading extremes.

    Given an initial shock (w0, r0) on the sphere, the Transformer generates
    the cascade event by event:
      1. Encode history -> h
      2. Sample direction W ~ vMF mixture (which asset follows)
      3. Sample magnitude R ~ TruncGamma(a, beta; R > u_tau(W))
      4. Sample timing dT ~ Exp(lambda) from Hawkes intensity
      5. Append event, repeat

    Stops only when T exceeds max_time (no fixed event count).

    If `prompt` is provided, it is used as historical context and the returned
    arrays include only the seed event (w0, r0) plus generated continuation,
    with time re-based to start at zero on the seed.
    """
    d_assets = model.d_assets
    seed_idx = 0

    if prompt is not None and len(prompt.get("W", [])) > 0:
        W = [np.asarray(w, dtype=np.float32) for w in prompt["W"]]
        R = [float(v) for v in prompt["R"]]
        T = [float(v) for v in prompt["T"]]
        dT_list = [float(v) for v in prompt.get("dT", np.array([], dtype=np.float32))]
        if len(W) != len(R) or len(W) != len(T):
            raise ValueError("Prompt context must have aligned W/R/T lengths.")
        if len(dT_list) != len(W):
            dT_arr = np.diff(np.array(T, dtype=np.float32), prepend=np.array(T, dtype=np.float32)[0])
            if len(dT_arr) > 1:
                med = float(np.median(dT_arr[1:]))
                dT_arr[0] = med if np.isfinite(med) and med > 1.0e-6 else 1.0
            elif len(dT_arr) == 1:
                dT_arr[0] = 1.0
            dT_list = dT_arr.astype(np.float32).tolist()

        recent_dt = np.array(dT_list[-min(len(dT_list), 64):], dtype=np.float32)
        recent_dt = recent_dt[np.isfinite(recent_dt) & (recent_dt > 1.0e-6)]
        dt_seed = float(np.median(recent_dt)) if recent_dt.size else 1.0
        t_seed = float(T[-1] + dt_seed)

        W.append(w0.astype(np.float32))
        R.append(float(r0))
        T.append(t_seed)
        dT_list.append(dt_seed)

        seed_idx = len(W) - 1
        horizon_end = t_seed + float(max_time)
    else:
        W = [w0.astype(np.float32)]
        R = [float(r0)]
        T = [0.0]
        dT_list = [1.0]  # median placeholder for first event
        horizon_end = float(max_time)

    for _ in range(max_events_cap):
        h, tokens_t, W_hist, R_hist, T_hist = _encode_history(model, W, R, dT_list, T, q_model=q_model)

        # 1. Sample next mark (direction + magnitude)
        w_next, r_next = _sample_candidate(model, h, R, q_model, d_assets, temperature=temperature, last_dt=dT_list[-1])

        # 2. Sample timing from Hawkes intensity: dT ~ Exp(lambda)
        lam, psi = _compute_intensity(model, h, T_hist, R_hist, W_hist)
        lam = max(lam, 1e-6)
        dt_next = float(np.random.exponential(1.0 / lam))
        t_next = T[-1] + dt_next

        if t_next > horizon_end:
            break

        W.append(w_next)
        R.append(r_next)
        dT_list.append(dt_next)
        T.append(t_next)

    if seed_idx > 0:
        T_out = np.array(T[seed_idx:], dtype=np.float32)
        dT_out = np.array(dT_list[seed_idx:], dtype=np.float32)
        W_out = np.array(W[seed_idx:], dtype=np.float32)
        R_out = np.array(R[seed_idx:], dtype=np.float32)

        T_out = T_out - T_out[0]
        if len(dT_out) > 1:
            d0 = float(np.median(dT_out[1:]))
            dT_out[0] = d0 if np.isfinite(d0) and d0 > 1.0e-6 else dT_out[0]
        elif len(dT_out) == 1:
            dT_out[0] = 1.0
    else:
        T_out = np.array(T, dtype=np.float32)
        dT_out = np.array(dT_list, dtype=np.float32)
        W_out = np.array(W, dtype=np.float32)
        R_out = np.array(R, dtype=np.float32)

    return {
        "T": T_out,
        "dT": dT_out,
        "W": W_out,
        "R": R_out,
    }


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
        h, tokens_t, W_hist, R_hist, T_hist = _encode_history(model, W, R, dT, T, q_model=q_model)
        lam_current, _ = _compute_intensity(model, h, T_hist, R_hist, W_hist)

        # Upper bound for thinning
        lam_star = max(safety_factor * lam_current, 1e-4)

        # Propose inter-arrival time
        dt_proposed = np.random.exponential(1.0 / lam_star)
        t_proposed = T[-1] + dt_proposed

        if max_time is not None and (t_proposed - t0) > max_time:
            break

        # Sample candidate mark
        w_candidate, r_candidate = _sample_candidate(model, h, R_hist, q_model, d_assets, last_dt=dT[-1] if dT else 1.0)

        # Temporarily add candidate to compute intensity at proposed time
        W_temp = W + [w_candidate]
        R_temp = R + [r_candidate]
        dT_temp = dT + [dt_proposed]
        T_temp = T + [t_proposed]

        h_temp, _, W_hist_temp, R_hist_temp, T_hist_temp = _encode_history(
            model, W_temp, R_temp, dT_temp, T_temp, q_model=q_model
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
    d_input = model.input_proj.in_features
    u_vals = None
    if d_input > 14:
        with torch.no_grad():
            u_vals = q_model(torch.tensor(events["W"], dtype=torch.float32)).numpy()
    tokens = build_tokens_from_arrays(events["W"], events["R"], events["dT"], u=u_vals)
    if tokens.shape[1] > d_input:
        tokens = tokens[:, :d_input]
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

    threshold_mode = cfg["extremes"].get("threshold_mode", "directional")
    if threshold_mode == "global":
        import json
        meta_path = Path(artifact_dir) / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        u_global = meta["u_global"]
        q_model = ConstantThreshold(u_global)
        print(f"[global threshold] u_tau = {u_global:.4f}")
    else:
        q_cfg = QuantileModelConfig(**cfg["extremes"]["quantile_model"])
        q_model = load_quantile_model(
            str(Path(artifact_dir) / "quantile_model.pt"),
            model.d_assets,
            q_cfg,
        )

    processed_dir = cfg.get("processed_dir", "data/processed_phase2")
    events_path = Path(processed_dir) / "events.npz"
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
