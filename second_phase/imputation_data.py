"""Build artifacts for generative return-series imputation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from second_phase.preprocess import compute_log_returns, fit_garch, standardize_laplace
from second_phase.utils import load_config


ROOT_DIR = Path(__file__).resolve().parents[1]
PHASE2_DIR = ROOT_DIR / "artifacts" / "phase2"
PRICES_PATH = ROOT_DIR / "data" / "raw" / "prices_1h_730d.csv"
X_SERIES_PATH = PHASE2_DIR / "x_series.parquet"
MAP_PATH = PHASE2_DIR / "x_to_returns_map.npz"


def _load_prices(symbols: list[str]) -> pd.DataFrame:
    if not PRICES_PATH.exists():
        raise FileNotFoundError(f"Prices file not found: {PRICES_PATH}")
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    missing = [s for s in symbols if s not in prices.columns]
    if missing:
        raise ValueError(f"Missing symbols in prices file: {missing}")
    return prices[symbols].dropna(how="any")


def build_x_and_returns_series(config_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config(config_path)
    symbols = [str(s) for s in cfg["data"]["symbols"]]
    prices = _load_prices(symbols)

    returns_dec = compute_log_returns(prices)
    residuals, _ = fit_garch(returns_dec, dist=cfg["preprocess"].get("garch_dist", "t"))
    x_df, _ = standardize_laplace(residuals, pit_clip=cfg["preprocess"].get("pit_clip", 1.0e-6))
    returns_pct = returns_dec * 100.0

    common = x_df.index.intersection(returns_pct.index)
    x_df = x_df.loc[common, symbols].dropna(how="any")
    returns_pct = returns_pct.loc[x_df.index, symbols].dropna(how="any")
    x_df = x_df.loc[returns_pct.index]
    if x_df.empty:
        raise RuntimeError("No aligned rows available for imputation artifacts.")
    return x_df, returns_pct


def save_imputation_artifacts(config_path: str, output_dir: str = str(PHASE2_DIR)) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_df, returns_pct = build_x_and_returns_series(config_path)
    x_series_path = out_dir / "x_series.parquet"
    map_path = out_dir / "x_to_returns_map.npz"

    x_df.to_parquet(x_series_path)

    symbols = list(x_df.columns)
    x_sorted = np.vstack([np.sort(x_df[s].to_numpy(dtype=np.float64)) for s in symbols])
    ret_sorted = np.vstack([np.sort(returns_pct[s].to_numpy(dtype=np.float64)) for s in symbols])
    np.savez(
        map_path,
        assets=np.array(symbols),
        x_sorted=x_sorted,
        ret_sorted=ret_sorted,
    )

    return {
        "x_series_path": str(x_series_path),
        "x_to_returns_map_path": str(map_path),
        "rows": str(len(x_df)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 2 imputation artifacts.")
    parser.add_argument("--config", default="configs/phase2.yaml")
    parser.add_argument("--output_dir", default=str(PHASE2_DIR))
    args = parser.parse_args()

    outputs = save_imputation_artifacts(args.config, args.output_dir)
    print("Saved imputation artifacts:")
    for key, value in outputs.items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
