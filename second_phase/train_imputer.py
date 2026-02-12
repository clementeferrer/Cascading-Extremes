"""Train SAITS imputer for dense generative return tracks."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from second_phase.imputation_data import save_imputation_artifacts


ROOT_DIR = Path(__file__).resolve().parents[1]
PHASE2_DIR = ROOT_DIR / "artifacts" / "phase2"
X_SERIES_PATH = PHASE2_DIR / "x_series.parquet"
IMPUTER_PATH = PHASE2_DIR / "imputer_saits.pypots"
IMPUTER_META_PATH = PHASE2_DIR / "imputer_meta.json"


def _load_x_series(config_path: str, output_dir: Path) -> pd.DataFrame:
    x_series_path = output_dir / X_SERIES_PATH.name
    if not x_series_path.exists():
        save_imputation_artifacts(config_path, str(output_dir))
    return pd.read_parquet(x_series_path).sort_index()


def _build_windows(x_df: pd.DataFrame, n_steps: int, stride: int) -> np.ndarray:
    arr = x_df.to_numpy(dtype=np.float32)
    n, d = arr.shape
    if n_steps <= 1:
        raise ValueError("n_steps must be > 1")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if n == 0:
        raise RuntimeError("x_series is empty")
    if n < n_steps:
        pad = np.repeat(arr[-1:, :], n_steps - n, axis=0)
        arr = np.concatenate([arr, pad], axis=0)
        n = arr.shape[0]

    starts = list(range(0, n - n_steps + 1, stride))
    tail_start = n - n_steps
    if not starts or starts[-1] != tail_start:
        starts.append(tail_start)
    windows = np.stack([arr[s : s + n_steps] for s in starts], axis=0)
    if windows.shape[0] < 2:
        windows = np.concatenate([windows, windows.copy()], axis=0)
    return windows


def _mask_windows(
    windows: np.ndarray,
    rng: np.random.Generator,
    point_mask_rate: float,
    block_mask_rate: float,
    block_min: int,
    block_max: int,
) -> np.ndarray:
    masked = windows.copy()
    n_samples, n_steps, d = masked.shape
    point_mask_rate = float(np.clip(point_mask_rate, 0.0, 0.95))
    block_mask_rate = float(np.clip(block_mask_rate, 0.0, 1.0))
    block_min = max(1, int(block_min))
    block_max = max(block_min, int(block_max))

    point_mask = rng.random(size=masked.shape) < point_mask_rate
    masked[point_mask] = np.nan

    for i in range(n_samples):
        if rng.random() > block_mask_rate:
            continue
        feat = int(rng.integers(0, d))
        block_len = int(rng.integers(block_min, block_max + 1))
        block_len = max(1, min(block_len, n_steps))
        start = int(rng.integers(0, n_steps - block_len + 1))
        masked[i, start : start + block_len, feat] = np.nan

    return masked


def _split_train_val(full_windows: np.ndarray, masked_windows: np.ndarray, val_ratio: float) -> Tuple[Dict, Dict]:
    n = full_windows.shape[0]
    n_val = max(1, int(round(n * val_ratio)))
    n_train = max(1, n - n_val)
    if n_train + n_val > n:
        n_val = n - n_train
    train_set = {
        "X": masked_windows[:n_train],
        "X_ori": full_windows[:n_train],
    }
    val_set = {
        "X": masked_windows[n_train : n_train + n_val],
        "X_ori": full_windows[n_train : n_train + n_val],
    }
    return train_set, val_set


def train_saits(
    config_path: str,
    output_dir: str = str(PHASE2_DIR),
    n_steps: int = 1024,
    stride: int = 24,
    epochs: int = 60,
    batch_size: int = 16,
    patience: int = 8,
    d_model: int = 128,
    n_layers: int = 2,
    n_heads: int = 4,
    d_ffn: int = 256,
    dropout: float = 0.1,
    attn_dropout: float = 0.1,
    point_mask_rate: float = 0.12,
    block_mask_rate: float = 0.85,
    block_min: int = 6,
    block_max: int = 48,
    val_ratio: float = 0.1,
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, str]:
    try:
        from pypots.imputation import SAITS  # type: ignore
    except Exception:
        try:
            from pypots.imputation.saits import SAITS  # type: ignore
        except Exception as exc:
            raise RuntimeError("PyPOTS is required to train the SAITS imputer.") from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_imputation_artifacts(config_path, str(out_dir))
    x_df = _load_x_series(config_path, out_dir)
    windows = _build_windows(x_df, n_steps=n_steps, stride=stride)

    rng = np.random.default_rng(seed)
    masked = _mask_windows(
        windows,
        rng=rng,
        point_mask_rate=point_mask_rate,
        block_mask_rate=block_mask_rate,
        block_min=block_min,
        block_max=block_max,
    )
    train_set, val_set = _split_train_val(windows, masked, val_ratio=val_ratio)

    n_features = windows.shape[-1]
    model_kwargs = {
        "n_steps": int(n_steps),
        "n_features": int(n_features),
        "n_layers": int(n_layers),
        "d_model": int(d_model),
        "n_heads": int(n_heads),
        "d_k": int(d_model // max(1, n_heads)),
        "d_v": int(d_model // max(1, n_heads)),
        "d_ffn": int(d_ffn),
        "dropout": float(dropout),
        "attn_dropout": float(attn_dropout),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "patience": int(patience),
        "num_workers": 0,
        "device": device,
        "verbose": True,
    }
    model = SAITS(**model_kwargs)
    model.fit(train_set=train_set, val_set=val_set)

    imputer_path = out_dir / IMPUTER_PATH.name
    model.save(str(imputer_path))

    meta = {
        "method": "saits",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {k: v for k, v in model_kwargs.items() if k not in {"verbose"}},
        "train": {
            "n_windows": int(windows.shape[0]),
            "seed": int(seed),
            "point_mask_rate": float(point_mask_rate),
            "block_mask_rate": float(block_mask_rate),
            "block_min": int(block_min),
            "block_max": int(block_max),
            "val_ratio": float(val_ratio),
        },
        "data": {
            "x_series_path": str(out_dir / X_SERIES_PATH.name),
            "x_to_returns_map_path": str(out_dir / "x_to_returns_map.npz"),
            "features": list(x_df.columns),
            "rows": int(len(x_df)),
        },
    }
    meta_path = out_dir / IMPUTER_META_PATH.name
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "imputer_path": str(imputer_path),
        "meta_path": str(meta_path),
        "x_series_path": str(out_dir / X_SERIES_PATH.name),
        "x_to_returns_map_path": str(out_dir / "x_to_returns_map.npz"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 2 SAITS imputer.")
    parser.add_argument("--config", default="configs/phase2.yaml")
    parser.add_argument("--output_dir", default=str(PHASE2_DIR))
    parser.add_argument("--n_steps", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_ffn", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--point_mask_rate", type=float, default=0.12)
    parser.add_argument("--block_mask_rate", type=float, default=0.85)
    parser.add_argument("--block_min", type=int, default=6)
    parser.add_argument("--block_max", type=int, default=48)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    outputs = train_saits(
        config_path=args.config,
        output_dir=args.output_dir,
        n_steps=args.n_steps,
        stride=args.stride,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ffn=args.d_ffn,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        point_mask_rate=args.point_mask_rate,
        block_mask_rate=args.block_mask_rate,
        block_min=args.block_min,
        block_max=args.block_max,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=args.device,
    )
    print("Saved SAITS imputer artifacts:")
    for key, value in outputs.items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
