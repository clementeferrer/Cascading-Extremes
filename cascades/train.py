import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from cascades.data import download as download_data
from cascades.dataset import EventSequenceDataset, SequenceConfig, build_events, compute_radial_angular
from cascades.extremes import QuantileModelConfig, fit_directional_threshold
from cascades.model import CascadingTransformer, ModelConfig
from cascades.preprocess import compute_log_returns, fit_garch, standardize
from cascades.utils import load_config, save_json, seed_all, ensure_dir


def split_events(events: Dict[str, np.ndarray], split: Tuple[float, float, float]) -> Tuple[Dict[str, np.ndarray], ...]:
    n = len(events["T"])
    n_train = int(n * split[0])
    n_val = int(n * split[1])
    idx_train = slice(0, n_train)
    idx_val = slice(n_train, n_train + n_val)
    idx_test = slice(n_train + n_val, n)

    def subset(idx):
        return {k: v[idx] for k, v in events.items()}

    return subset(idx_train), subset(idx_val), subset(idx_test)


def train_loop(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    count = 0
    for batch in loader:
        tokens = batch["tokens"].to(device)
        W = batch["W"].to(device)
        R = batch["R"].to(device)
        u = batch["u"].to(device)
        dT = batch["dT"].to(device)
        T = batch["T"].to(device)

        tokens_in = tokens[:, :-1, :]
        W_next = W[:, 1:, :]
        R_next = R[:, 1:]
        u_next = u[:, 1:]
        dT_next = dT[:, 1:]
        T_in = T[:, :-1]
        R_in = R[:, :-1]

        out = model.log_likelihood(tokens_in, W_next, R_next, dT_next, T_in, R_in, u_next)
        loglik = out["log_p_w"] + out["log_p_r"] + out["log_p_t"]
        loss = -loglik.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    return total_loss / max(count, 1)


def eval_loop(model, loader, device):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device)
            W = batch["W"].to(device)
            R = batch["R"].to(device)
            u = batch["u"].to(device)
            dT = batch["dT"].to(device)
            T = batch["T"].to(device)

            tokens_in = tokens[:, :-1, :]
            W_next = W[:, 1:, :]
            R_next = R[:, 1:]
            u_next = u[:, 1:]
            dT_next = dT[:, 1:]
            T_in = T[:, :-1]
            R_in = R[:, :-1]

            out = model.log_likelihood(tokens_in, W_next, R_next, dT_next, T_in, R_in, u_next)
            loglik = out["log_p_w"] + out["log_p_r"] + out["log_p_t"]
            loss = -loglik.mean()

            total_loss += loss.item()
            count += 1

    return total_loss / max(count, 1)


def fit(cfg: Dict, events: Optional[Dict[str, np.ndarray]] = None):
    seed_all(cfg["train"]["seed"])

    if events is None:
        data_cfg = cfg["data"]
        prices = download_data(
            data_cfg["symbols"],
            period=data_cfg.get("period", "730d"),
            interval=data_cfg.get("interval", "1h"),
            cache_dir=data_cfg.get("cache_dir", "data/raw"),
            price_field=data_cfg.get("price_field", "Close"),
        )

        returns = compute_log_returns(prices)
        residuals, sigma = fit_garch(returns, dist=cfg["preprocess"].get("garch_dist", "t"))
        X, cdfs = standardize(residuals, pit_clip=cfg["preprocess"].get("pit_clip", 1.0e-6))
        X = X.dropna()
        timestamps = X.index

        R_all, W_all = compute_radial_angular(X.values, eps=cfg["extremes"].get("eps", 1.0e-8))
        q_cfg = QuantileModelConfig(**cfg["extremes"]["quantile_model"])
        u_tau_fn, u_meta, q_model = fit_directional_threshold(
            W_all,
            R_all,
            tau=cfg["extremes"]["tau"],
            config=q_cfg,
            device=cfg["train"].get("device", "cpu"),
        )

        events = build_events(X, timestamps, u_tau_fn, eps=cfg["extremes"].get("eps", 1.0e-8))

        processed_dir = ensure_dir("data/processed")
        np.savez(
            Path(processed_dir) / "events.npz",
            **events,
        )

        np.savez(
            "artifacts/cdfs.npz",
            **{k: v.sorted_values for k, v in cdfs.items()},
        )
        torch.save(q_model.state_dict(), "artifacts/quantile_model.pt")
        save_json(
            "artifacts/meta.json",
            {
                "symbols": list(prices.columns),
                "tau": cfg["extremes"]["tau"],
                "config": cfg,
                "u_meta": {"tau": float(u_meta["tau"][0])},
            },
        )

    split = tuple(cfg["dataset"]["split"])
    train_events, val_events, test_events = split_events(events, split)

    seq_len = cfg["dataset"]["seq_len"]
    min_len = 16
    n_train = len(train_events["T"])
    n_val = len(val_events["T"])
    if n_train < seq_len or n_val < seq_len:
        new_len = min(seq_len, n_train, n_val)
        if new_len < min_len:
            new_len = max(4, new_len)
        print(f"[warn] Adjusting seq_len from {seq_len} to {new_len} to ensure non-empty splits.")
        seq_len = new_len

    seq_cfg = SequenceConfig(seq_len=seq_len, stride=cfg["dataset"]["stride"])

    train_ds = EventSequenceDataset(train_events, seq_cfg)
    val_ds = EventSequenceDataset(val_events, seq_cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    d_assets = events["W"].shape[1]
    d_input = events["tokens"].shape[1]
    model_cfg = ModelConfig(**cfg["model"])
    model = CascadingTransformer(d_input=d_input, d_assets=d_assets, cfg=model_cfg)
    device = torch.device(cfg["train"].get("device", "cpu"))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    best_val = float("inf")
    metrics = []
    for epoch in range(cfg["train"]["epochs"]):
        train_loss = train_loop(model, train_loader, optimizer, device)
        val_loss = eval_loop(model, val_loader, device)
        metrics.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            ensure_dir("artifacts")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_cfg": asdict(model_cfg),
                    "d_input": d_input,
                    "d_assets": d_assets,
                },
                "artifacts/model.pt",
            )

    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Cascading Extremes Transformer.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, metrics = fit(cfg)
    print("Training complete. Final epoch:", metrics[-1] if metrics else "n/a")


if __name__ == "__main__":
    main()
