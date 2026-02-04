import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from cascades.dataset import zeta
from cascades.extremes import DirectionalQuantileMLP, QuantileModelConfig
from cascades.model import CascadingTransformer, ModelConfig
from cascades.utils import load_config, ensure_dir


def load_model(path: str) -> CascadingTransformer:
    payload = torch.load(path, map_location="cpu")
    cfg = ModelConfig(**payload["model_cfg"])
    model = CascadingTransformer(d_input=payload["d_input"], d_assets=payload["d_assets"], cfg=cfg)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def load_quantile_model(path: str, d_assets: int, cfg: QuantileModelConfig) -> DirectionalQuantileMLP:
    model = DirectionalQuantileMLP(d_assets, cfg.hidden_sizes)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def sample_dirichlet_mixture(pi: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    comp = torch.distributions.Categorical(pi).sample()
    alpha_k = alpha[comp]
    dist = torch.distributions.Dirichlet(alpha_k)
    return dist.sample()


def sample_trunc_gamma(shape: float, rate: float, u: float, max_tries: int = 512) -> float:
    dist = torch.distributions.Gamma(torch.tensor(shape), torch.tensor(rate))
    last = u
    for _ in range(max_tries):
        r = dist.sample().item()
        last = r
        if r > u:
            return r
    return max(last, u + 1.0e-6)


def build_token(w: np.ndarray, r: float, dt: float) -> np.ndarray:
    z = zeta(w[None, :])[0]
    return np.concatenate([w, z, [np.log(r + 1.0e-8), np.log(dt + 1.0e-8)]], axis=0)


def generate(seed: Dict[str, np.ndarray], horizon: int, model: CascadingTransformer, q_model: DirectionalQuantileMLP) -> Dict[str, np.ndarray]:
    W = seed["W"].tolist()
    R = seed["R"].tolist()
    dT = seed["dT"].tolist()
    T = seed["T"].tolist()

    max_len = model.cfg.max_len

    for _ in range(horizon):
        start = max(0, len(W) - max_len)
        W_hist = np.array(W[start:], dtype=np.float32)
        R_hist = np.array(R[start:], dtype=np.float32)
        dT_hist = np.array(dT[start:], dtype=np.float32)
        T_hist = np.array(T[start:], dtype=np.float32)

        tokens = np.stack([build_token(W_hist[i], R_hist[i], dT_hist[i]) for i in range(len(W_hist))], axis=0)
        tokens_t = torch.tensor(tokens[None, :, :], dtype=torch.float32)
        with torch.no_grad():
            h = model.encode(tokens_t)
            pi_all, alpha_all = model.dirichlet_params(h)
            pi = pi_all[0, -1]
            alpha = alpha_all[0, -1]
            w_next = sample_dirichlet_mixture(pi, alpha).numpy()

            log_r_in = torch.tensor([[np.log(R_hist[-1] + 1.0e-8)]], dtype=torch.float32)
            h_last = h[:, -1:, :]
            rate = model.gauge_rate(h_last, log_r_in, torch.tensor(w_next[None, None, :], dtype=torch.float32)).item()
            u_next = q_model(torch.tensor(w_next[None, :], dtype=torch.float32)).item()
            shape = model.softplus(model.a_param).item() + model.cfg.a_min
            r_next = sample_trunc_gamma(shape, rate, u_next)

            T_tensor = torch.tensor(T_hist[None, :], dtype=torch.float32)
            R_tensor = torch.tensor(R_hist[None, :], dtype=torch.float32)
            log_r_hist = torch.log(R_tensor + 1.0e-8)
            lam, _ = model.hawkes_intensity(h, T_tensor, log_r_hist)
            lam_last = lam[0, -1].item()
            dt_next = np.random.exponential(1.0 / max(lam_last, 1.0e-8))

        T.append(T[-1] + dt_next)
        dT.append(dt_next)
        W.append(w_next)
        R.append(r_next)

    return {
        "T": np.array(T, dtype=np.float32),
        "dT": np.array(dT, dtype=np.float32),
        "W": np.array(W, dtype=np.float32),
        "R": np.array(R, dtype=np.float32),
    }


def generate_with_limits(
    seed: Dict[str, np.ndarray],
    max_events: int,
    max_time: Optional[float],
    model: CascadingTransformer,
    q_model: DirectionalQuantileMLP,
) -> Dict[str, np.ndarray]:
    W = seed["W"].tolist()
    R = seed["R"].tolist()
    dT = seed["dT"].tolist()
    T = seed["T"].tolist()

    max_len = model.cfg.max_len
    t0 = T[-1] if T else 0.0

    for _ in range(max_events):
        start = max(0, len(W) - max_len)
        W_hist = np.array(W[start:], dtype=np.float32)
        R_hist = np.array(R[start:], dtype=np.float32)
        dT_hist = np.array(dT[start:], dtype=np.float32)
        T_hist = np.array(T[start:], dtype=np.float32)

        tokens = np.stack([build_token(W_hist[i], R_hist[i], dT_hist[i]) for i in range(len(W_hist))], axis=0)
        tokens_t = torch.tensor(tokens[None, :, :], dtype=torch.float32)
        with torch.no_grad():
            h = model.encode(tokens_t)
            pi_all, alpha_all = model.dirichlet_params(h)
            pi = pi_all[0, -1]
            alpha = alpha_all[0, -1]
            w_next = sample_dirichlet_mixture(pi, alpha).numpy()

            log_r_in = torch.tensor([[np.log(R_hist[-1] + 1.0e-8)]], dtype=torch.float32)
            h_last = h[:, -1:, :]
            rate = model.gauge_rate(h_last, log_r_in, torch.tensor(w_next[None, None, :], dtype=torch.float32)).item()
            u_next = q_model(torch.tensor(w_next[None, :], dtype=torch.float32)).item()
            shape = model.softplus(model.a_param).item() + model.cfg.a_min
            r_next = sample_trunc_gamma(shape, rate, u_next)

            T_tensor = torch.tensor(T_hist[None, :], dtype=torch.float32)
            R_tensor = torch.tensor(R_hist[None, :], dtype=torch.float32)
            log_r_hist = torch.log(R_tensor + 1.0e-8)
            lam, _ = model.hawkes_intensity(h, T_tensor, log_r_hist)
            lam_last = lam[0, -1].item()
            dt_next = np.random.exponential(1.0 / max(lam_last, 1.0e-8))

        next_t = T[-1] + dt_next
        if max_time is not None and (next_t - t0) > max_time:
            break

        T.append(next_t)
        dT.append(dt_next)
        W.append(w_next)
        R.append(r_next)

    return {
        "T": np.array(T, dtype=np.float32),
        "dT": np.array(dT, dtype=np.float32),
        "W": np.array(W, dtype=np.float32),
        "R": np.array(R, dtype=np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate cascades from trained model.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = load_model("artifacts/model.pt")
    q_cfg = QuantileModelConfig(**cfg["extremes"]["quantile_model"])
    q_model = load_quantile_model("artifacts/quantile_model.pt", model.d_assets, q_cfg)

    events = np.load(Path("data/processed") / "events.npz")
    seed_len = cfg["simulate"]["seed_len"]
    seed = {k: events[k][-seed_len:] for k in events.files}

    sim = generate(seed, cfg["simulate"]["horizon_events"], model, q_model)
    ensure_dir("artifacts")
    np.savez("artifacts/simulated_events.npz", **sim)
    print("Simulation complete -> artifacts/simulated_events.npz")


if __name__ == "__main__":
    main()
