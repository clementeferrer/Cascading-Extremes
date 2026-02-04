import argparse
from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf

from cascades.utils import ensure_dir, load_config


def download(
    symbols: List[str],
    period: str = "730d",
    interval: str = "1h",
    cache_dir: str = "data/raw",
    price_field: str = "Close",
) -> pd.DataFrame:
    ensure_dir(cache_dir)
    cache_path = Path(cache_dir) / f"prices_{interval}_{period.replace(' ', '')}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    df = yf.download(
        tickers=" ".join(symbols),
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    if isinstance(df.columns, pd.MultiIndex):
        if price_field not in df.columns.get_level_values(0):
            raise ValueError(f"Price field {price_field} not in data columns")
        df = df[price_field]
    else:
        df = df[[price_field]].rename(columns={price_field: symbols[0]})

    df = df.dropna(how="all")
    df.to_csv(cache_path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Download hourly crypto prices.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    symbols = data_cfg["symbols"]
    period = data_cfg.get("period", "730d")
    interval = data_cfg.get("interval", "1h")
    cache_dir = data_cfg.get("cache_dir", "data/raw")
    price_field = data_cfg.get("price_field", "Close")

    df = download(symbols, period=period, interval=interval, cache_dir=cache_dir, price_field=price_field)
    print(f"Downloaded data: {df.shape} -> {cache_dir}")


if __name__ == "__main__":
    main()
