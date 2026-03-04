"""Training pipeline for VAR(1) simulated data.

Full pipeline: VAR(1) → GARCH → PIT → Laplace → threshold → events → train model
→ simulate → genealogy.

Run from project root:
    python -m streamlit_test.train_var --config configs/phase2_var.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure project root is importable
ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from second_phase.dataset import build_events_sphere, compute_radial_angular_l2
from second_phase.extremes import QuantileModelConfig, fit_sphere_threshold
from second_phase.preprocess import fit_garch, standardize_laplace
from second_phase.simulate import generate_with_genealogy
from second_phase.train import fit
from second_phase.utils import ensure_dir, load_config, save_json, seed_all
from streamlit_test.var_generate import simulate_var1


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 2 model on VAR(1) simulated data.")
    parser.add_argument("--config", default="configs/phase2_var.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_all(cfg["train"]["seed"])

    artifact_dir = cfg.get("artifact_dir", "artifacts/phase2_var")
    processed_dir = "data/processed_phase2_var"
    ensure_dir(artifact_dir)
    ensure_dir(processed_dir)

    var_cfg = cfg.get("var", {})
    symbols = cfg["data"]["symbols"]

    # ── 1. Generate VAR(1) returns ─────────────────────────────────────
    print("[1/9] Generating VAR(1) returns ...")
    A = np.array(var_cfg["A"], dtype=np.float64) if "A" in var_cfg else None
    Sigma = np.array(var_cfg["Sigma"], dtype=np.float64) if "Sigma" in var_cfg else None

    returns = simulate_var1(
        n_obs=var_cfg.get("n_obs", 17_520),
        d=var_cfg.get("d", len(symbols)),
        A=A,
        Sigma=Sigma,
        df=var_cfg.get("df", 3.0),
        seed=var_cfg.get("seed", 42),
        symbols=symbols,
    )
    print(f"   {returns.shape[0]} observations, {returns.shape[1]} assets")

    # ── 2. Fit GARCH ──────────────────────────────────────────────────
    print("[2/9] Fitting GARCH ...")
    residuals, sigma = fit_garch(returns, dist=cfg["preprocess"].get("garch_dist", "t"))

    # ── 3. Standardize to Laplace margins ─────────────────────────────
    print("[3/9] Standardizing to Laplace margins ...")
    X, cdfs = standardize_laplace(residuals, pit_clip=cfg["preprocess"].get("pit_clip", 1e-6))
    X = X.dropna()
    timestamps = X.index
    print(f"   {len(X)} valid observations after dropna")

    # ── 4. Radial-angular decomposition ───────────────────────────────
    print("[4/9] Computing radial-angular decomposition ...")
    R_all, W_all = compute_radial_angular_l2(X.values, eps=cfg["extremes"].get("eps", 1e-8))

    # ── 5. Fit directional threshold ──────────────────────────────────
    print("[5/9] Fitting directional threshold ...")
    q_cfg = QuantileModelConfig(**cfg["extremes"]["quantile_model"])
    device = cfg["train"].get("device", "cpu")
    u_tau_fn, u_meta, q_model = fit_sphere_threshold(
        W_all, R_all,
        tau=cfg["extremes"]["tau"],
        config=q_cfg,
        device=device,
    )

    # ── 6. Build extreme events ───────────────────────────────────────
    print("[6/9] Building extreme events ...")
    events = build_events_sphere(X, timestamps, u_tau_fn, eps=cfg["extremes"].get("eps", 1e-8))
    n_events = len(events["T"])
    print(f"   {n_events} extreme events (tau={cfg['extremes']['tau']})")

    # ── 7. Save pre-training artifacts ────────────────────────────────
    print("[7/9] Saving pre-training artifacts ...")
    np.savez(Path(processed_dir) / "events.npz", **events)
    np.savez(
        Path(artifact_dir) / "cdfs.npz",
        **{k: v.sorted_values for k, v in cdfs.items()},
    )
    torch.save(q_model.state_dict(), Path(artifact_dir) / "quantile_model.pt")
    save_json(
        str(Path(artifact_dir) / "meta.json"),
        {
            "symbols": symbols,
            "tau": cfg["extremes"]["tau"],
            "config": cfg,
            "u_meta": {"tau": float(u_meta["tau"][0])},
            "source": "var1",
            "var": {
                "n_obs": int(returns.shape[0]),
                "df": var_cfg.get("df", 3.0),
                "seed": var_cfg.get("seed", 42),
            },
        },
    )

    # Save bulk observations (all R * W positions, not just extremes)
    positions = (R_all[:, None] * W_all).astype(np.float32)
    np.savez(Path(artifact_dir) / "bulk_observations.npz", positions=positions)

    # ── 8. Train model ────────────────────────────────────────────────
    print("[8/9] Training model ...")
    model, metrics = fit(cfg, events=events)
    print(f"   Training complete. Final epoch: {metrics[-1] if metrics else 'n/a'}")

    # ── 9. Simulate + genealogy ───────────────────────────────────────
    print("[9/9] Running simulation + genealogy ...")
    from second_phase.simulate import load_model, load_quantile_model

    model = load_model(str(Path(artifact_dir) / "model.pt"))
    q_model_loaded = load_quantile_model(
        str(Path(artifact_dir) / "quantile_model.pt"),
        model.d_assets,
        q_cfg,
    )

    seed_len = cfg["simulate"]["seed_len"]
    seed = {k: events[k][-seed_len:] for k in ["W", "R", "dT", "T"]}

    sim_events, genealogy = generate_with_genealogy(
        seed,
        max_events=cfg["simulate"]["horizon_events"],
        max_time=cfg["simulate"].get("max_time"),
        model=model,
        q_model=q_model_loaded,
        safety_factor=cfg["simulate"].get("safety_factor", 1.5),
    )

    np.savez(str(Path(artifact_dir) / "simulated_events.npz"), **sim_events)
    np.savez(
        str(Path(artifact_dir) / "genealogy.npz"),
        parents=genealogy.parents,
        immigrant_mask=genealogy.immigrant_mask,
        cascade_probs=genealogy.cascade_probs,
    )

    print(f"\nAll artifacts saved to {artifact_dir}/")
    print(f"  Events:     {n_events}")
    print(f"  Simulated:  {len(sim_events['T'])}")
    print(f"  Immigrants: {genealogy.immigrant_mask.sum()} / {len(genealogy.parents)}")
    print(f"  Clusters:   {len(genealogy.clusters)}")


if __name__ == "__main__":
    main()
