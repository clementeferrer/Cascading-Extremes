from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from cascades.dataset import zeta
from cascades.extremes import DirectionalQuantileMLP, QuantileModelConfig
from cascades.model import CascadingTransformer, ModelConfig
from cascades.utils import ensure_dir, load_config
from cascades.viz_export.schema import RunMeta


def _config_hash(path: str) -> str:
    data = Path(path).read_bytes()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    arr = np.load(path)
    return {k: arr[k] for k in arr.files}


def _load_quantile_model(cfg, d_assets: int, device: str = "cpu") -> Optional[DirectionalQuantileMLP]:
    model_path = Path("artifacts") / "quantile_model.pt"
    if not model_path.exists():
        return None
    q_cfg = QuantileModelConfig(**cfg["extremes"]["quantile_model"])
    model = DirectionalQuantileMLP(d_assets, q_cfg.hidden_sizes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def _compute_u_tau(source: str, events: Dict[str, np.ndarray], q_model: Optional[DirectionalQuantileMLP]) -> np.ndarray:
    if source == "real":
        if "u" not in events:
            raise ValueError("Real events missing u threshold")
        return events["u"].astype(np.float32)
    if q_model is None:
        # fallback: global quantile
        r = events["R"].astype(np.float32)
        return np.full_like(r, np.quantile(r, 0.98), dtype=np.float32)
    with torch.no_grad():
        w = torch.tensor(events["W"], dtype=torch.float32)
        u = q_model(w).cpu().numpy().astype(np.float32)
    return u


def _compute_tokens(W: np.ndarray, R: np.ndarray, dT: np.ndarray) -> np.ndarray:
    z = zeta(W)
    return np.concatenate([W, z, np.log(R + 1.0e-8)[:, None], np.log(dT + 1.0e-8)[:, None]], axis=1)


def _compute_intensity(events: Dict[str, np.ndarray], model_path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not model_path.exists():
        return None
    payload = torch.load(model_path, map_location="cpu")
    cfg = ModelConfig(**payload["model_cfg"])
    model = CascadingTransformer(d_input=payload["d_input"], d_assets=payload["d_assets"], cfg=cfg)
    model.load_state_dict(payload["model_state"])
    model.eval()

    T = events["T"].astype(np.float32)
    R = events["R"].astype(np.float32)
    W = events["W"].astype(np.float32)
    dT = events.get("dT")
    if dT is None:
        dT = np.diff(T, prepend=T[0]).astype(np.float32)
        dT[0] = np.median(dT[1:]) if len(dT) > 1 else 1.0
    tokens = _compute_tokens(W, R, dT)

    seq_len = tokens.shape[0]
    max_len = model.cfg.max_len

    lam_out = np.full(seq_len, np.nan, dtype=np.float32)
    psi_out = np.full(seq_len, np.nan, dtype=np.float32)

    if seq_len <= max_len:
        tokens_t = torch.tensor(tokens[None, :, :], dtype=torch.float32)
        T_tensor = torch.tensor(T[None, :], dtype=torch.float32)
        R_tensor = torch.tensor(R[None, :], dtype=torch.float32)
        log_r = torch.log(R_tensor + 1.0e-8)

        with torch.no_grad():
            h = model.encode(tokens_t)
            lam, psi = model.hawkes_intensity(h, T_tensor, log_r)

        lam_out[:] = lam.squeeze(0).cpu().numpy().astype(np.float32)
        psi_out[:] = psi.squeeze(0).cpu().numpy().astype(np.float32)
    else:
        # Chunked approximation: intensities computed within each window of max_len.
        for start in range(0, seq_len, max_len):
            end = min(seq_len, start + max_len)
            tokens_chunk = torch.tensor(tokens[start:end][None, :, :], dtype=torch.float32)
            T_chunk = torch.tensor(T[start:end][None, :], dtype=torch.float32)
            R_chunk = torch.tensor(R[start:end][None, :], dtype=torch.float32)
            log_r = torch.log(R_chunk + 1.0e-8)

            with torch.no_grad():
                h = model.encode(tokens_chunk)
                lam, psi = model.hawkes_intensity(h, T_chunk, log_r)

            lam_out[start:end] = lam.squeeze(0).cpu().numpy().astype(np.float32)
            psi_out[start:end] = psi.squeeze(0).cpu().numpy().astype(np.float32)

    mu_out = lam_out - psi_out
    return {"lambda": lam_out, "psi": psi_out, "mu": mu_out}


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    if len(arr) == 0:
        return arr
    window = max(1, min(window, len(arr)))
    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = arr[start : i + 1].mean()
    return out


def export_run_from_arrays(
    config_path: str,
    run_id: str,
    source: str,
    arrays: Dict[str, np.ndarray],
    output_dir: str = "artifacts/runs",
) -> Path:
    cfg = load_config(config_path)
    assets = cfg["data"]["symbols"]

    T = arrays["T"].astype(np.float32)
    R = arrays["R"].astype(np.float32)
    W = arrays["W"].astype(np.float32)
    dT = arrays.get("dT")
    if dT is None:
        dT = np.diff(T, prepend=T[0]).astype(np.float32)
        dT[0] = np.median(dT[1:]) if len(dT) > 1 else 1.0

    q_model = _load_quantile_model(cfg, W.shape[1])
    if "u_tau" in arrays:
        u_tau = arrays["u_tau"].astype(np.float32)
    else:
        u_tau = _compute_u_tau(source, {"T": T, "R": R, "W": W}, q_model)

    model_path = Path("artifacts") / "model.pt"
    intensity = _compute_intensity({"T": T, "R": R, "W": W, "dT": dT}, model_path)

    dominant_idx = np.argmax(W, axis=1)
    asset_names = [assets[i] if i < len(assets) else f"asset_{i}" for i in dominant_idx]

    df_events = pd.DataFrame(
        {
            "id": np.arange(len(T), dtype=np.int32),
            "t": T,
            "w": W.tolist(),
            "mag": R,
            "u_tau": u_tau,
            "asset": asset_names,
            "intensity": intensity["lambda"] if intensity is not None else np.nan,
            "parent_id": pd.Series([None] * len(T), dtype="object"),
            "is_real": source == "real",
        }
    )

    mean_mag = _rolling_mean(R, window=20)
    event_rate = 1.0 / np.maximum(_rolling_mean(dT, window=20), 1.0e-6)
    per_asset_counts = []
    counts = {a: 0 for a in assets}
    for a in asset_names:
        counts[a] = counts.get(a, 0) + 1
        per_asset_counts.append(json.dumps(counts))

    df_metrics = pd.DataFrame(
        {
            "t": T,
            "lambda": intensity["lambda"] if intensity is not None else np.nan,
            "psi": intensity["psi"] if intensity is not None else np.nan,
            "mu": intensity["mu"] if intensity is not None else np.nan,
            "mean_mag": mean_mag,
            "event_rate": event_rate,
            "per_asset_counts": per_asset_counts,
            "direction_density_bin": [None] * len(T),
        }
    )

    out_dir = Path(output_dir) / run_id
    ensure_dir(str(out_dir))

    pq.write_table(pa.Table.from_pandas(df_events), out_dir / "events.parquet")
    pq.write_table(pa.Table.from_pandas(df_metrics), out_dir / "metrics.parquet")

    meta = RunMeta(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        source=source,
        assets=assets,
        freq=cfg["data"].get("interval", "1h"),
        threshold={"tau": cfg["extremes"]["tau"], "model": "directional_quantile_mlp"},
        model_checkpoint=str(model_path) if model_path.exists() else None,
        config_hash=_config_hash(config_path),
    )

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta.model_dump(), f, indent=2)

    index_path = Path(output_dir) / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {"runs": []}

    index["runs"] = [r for r in index["runs"] if r.get("run_id") != run_id]
    index["runs"].append(
        {
            "run_id": run_id,
            "created_at": meta.created_at,
            "source": source,
            "assets": assets,
            "path": str(out_dir),
        }
    )

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    return out_dir


def export_run(config_path: str, run_id: str, source: str, output_dir: str = "artifacts/runs") -> Path:
    if source == "real":
        events_path = Path("data/processed") / "events.npz"
    else:
        events_path = Path("artifacts") / "simulated_events.npz"

    if not events_path.exists():
        raise FileNotFoundError(f"Events not found: {events_path}")

    events = _load_npz(events_path)
    arrays = {
        "T": events["T"],
        "R": events["R"],
        "W": events["W"],
        "dT": events.get("dT"),
    }
    if "u" in events:
        arrays["u_tau"] = events["u"]
    return export_run_from_arrays(config_path, run_id, source, arrays, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export cascading extremes run artifacts.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--source", choices=["real", "simulated", "generative"], required=True)
    parser.add_argument("--output_dir", default="artifacts/runs")
    args = parser.parse_args()

    out = export_run(args.config, args.run_id, args.source, args.output_dir)
    print(f"Exported run to {out}")


if __name__ == "__main__":
    main()
