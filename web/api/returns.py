"""Run-aligned return series for immersive viewer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from web.api.storage import get_run_path, read_meta
except ModuleNotFoundError:
    from storage import get_run_path, read_meta  # type: ignore

try:
    from web.api.generative_imputation import artifact_signature, build_generative_payload
except ModuleNotFoundError:
    from generative_imputation import artifact_signature, build_generative_payload  # type: ignore


ROOT_DIR = Path(__file__).resolve().parents[2]
PRICES_PATH = ROOT_DIR / "data" / "raw" / "prices_1h_730d.csv"


@dataclass
class ReturnsError(Exception):
    status_code: int
    detail: str

    def __str__(self) -> str:
        return self.detail


_cached_payloads: Dict[str, Tuple[Tuple[int, ...], Dict]] = {}


def _mtime_ns(path: Path) -> int:
    if not path.exists():
        return -1
    return path.stat().st_mtime_ns


def _normalize_asset_name(asset: str) -> str:
    return asset.strip().upper().replace("_", "-")


def _asset_lookup(asset_names: List[str]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for name in asset_names:
        norm = _normalize_asset_name(name)
        lookup[norm] = name
        if norm.endswith("-USD"):
            lookup[norm[:-4]] = name
    return lookup


def _resolve_event_asset(event_asset: str, lookup: Dict[str, str]) -> str | None:
    norm = _normalize_asset_name(event_asset)
    if norm in lookup:
        return lookup[norm]
    if norm.endswith("-USD") and norm[:-4] in lookup:
        return lookup[norm[:-4]]
    with_usd = f"{norm}-USD"
    if with_usd in lookup:
        return lookup[with_usd]
    return None


def _load_returns_pct(assets: List[str]) -> pd.DataFrame:
    if not PRICES_PATH.exists():
        raise ReturnsError(status_code=500, detail=f"Prices file not found: {PRICES_PATH}")
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    missing = [a for a in assets if a not in prices.columns]
    if missing:
        raise ReturnsError(status_code=500, detail=f"Missing price columns for assets: {missing}")
    returns = np.log(prices[assets]).diff().dropna(how="all") * 100.0
    if returns.empty:
        raise ReturnsError(status_code=500, detail="No return observations available from price history.")
    return returns


def _align_offset(event_t: np.ndarray, event_asset_idx: np.ndarray, returns_arr: np.ndarray) -> Tuple[int, int]:
    max_event_t = int(event_t.max())
    candidate_end = returns_arr.shape[0] - 1 - max_event_t
    if candidate_end < 0:
        return 0, 0

    best_offset = 0
    best_score = -1.0
    for offset in range(candidate_end + 1):
        score = float(np.abs(returns_arr[offset + event_t, event_asset_idx]).sum())
        if score > best_score:
            best_score = score
            best_offset = offset
    return best_offset, candidate_end + 1


def _empty_real_payload(run_id: str, assets: List[str]) -> Dict:
    return {
        "run_id": run_id,
        "units": "log_return_pct",
        "assets": assets,
        "series": {a: [] for a in assets},
        "extreme_points": {a: [] for a in assets},
        "alignment": {
            "method": "offset_estimated_max_abs_return",
            "offset_hours": 0,
            "candidate_count": 0,
            "start_datetime_utc": None,
        },
        "count": 0,
        "series_mode": "real_dense",
    }


def _build_real_payload(run_id: str, assets: List[str], events_path: Path) -> Dict:
    events_df = pd.read_parquet(events_path, columns=["t", "asset"])
    if events_df.empty:
        return _empty_real_payload(run_id, assets)

    returns_df = _load_returns_pct(assets)
    returns_arr = returns_df.to_numpy(dtype=np.float64)
    asset_to_idx = {a: i for i, a in enumerate(assets)}
    event_lookup = _asset_lookup(assets)

    event_pairs: List[Tuple[int, str]] = []
    for t_val, asset_val in zip(events_df["t"].tolist(), events_df["asset"].tolist()):
        t_i = int(round(float(t_val)))
        if t_i < 0:
            continue
        resolved_asset = _resolve_event_asset(str(asset_val), event_lookup)
        if resolved_asset is None:
            continue
        event_pairs.append((t_i, resolved_asset))

    if not event_pairs:
        return _empty_real_payload(run_id, assets)

    event_t = np.array([pair[0] for pair in event_pairs], dtype=np.int64)
    event_asset_idx = np.array([asset_to_idx[pair[1]] for pair in event_pairs], dtype=np.int64)
    offset, candidate_count = _align_offset(event_t, event_asset_idx, returns_arr)

    max_t = int(event_t.max())
    series: Dict[str, List[List[float]]] = {}
    extreme_points: Dict[str, List[List[float]]] = {a: [] for a in assets}

    for asset, col_idx in asset_to_idx.items():
        values = returns_arr[offset : offset + max_t + 1, col_idx]
        series[asset] = [[float(i), float(v)] for i, v in enumerate(values) if np.isfinite(v)]

    for t_i, asset in event_pairs:
        idx = offset + t_i
        col_idx = asset_to_idx[asset]
        if idx < 0 or idx >= returns_arr.shape[0]:
            continue
        val = returns_arr[idx, col_idx]
        if np.isfinite(val):
            extreme_points[asset].append([float(t_i), float(val)])

    return {
        "run_id": run_id,
        "units": "log_return_pct",
        "assets": assets,
        "series": series,
        "extreme_points": extreme_points,
        "alignment": {
            "method": "offset_estimated_max_abs_return",
            "offset_hours": int(offset),
            "candidate_count": int(candidate_count),
            "start_datetime_utc": pd.Timestamp(returns_df.index[offset]).isoformat(),
        },
        "count": int(len(event_pairs)),
        "series_mode": "real_dense",
    }


def _cache_signature(meta: Dict, events_path: Path, meta_path: Path) -> Tuple[int, ...]:
    source = str(meta.get("source") or "")
    if source == "real":
        return (_mtime_ns(events_path), _mtime_ns(PRICES_PATH))
    if source == "generative":
        imputer_sig = artifact_signature()
        return (_mtime_ns(events_path), _mtime_ns(meta_path), *imputer_sig)
    return (_mtime_ns(events_path), _mtime_ns(meta_path))


def get_returns_payload(run_id: str) -> Dict:
    run_path = get_run_path(run_id)
    meta_path = run_path / "meta.json"
    events_path = run_path / "events.parquet"

    if not meta_path.exists():
        raise ReturnsError(status_code=404, detail="Run not found")
    if not events_path.exists():
        raise ReturnsError(status_code=404, detail="Events not found for run")

    meta = read_meta(run_id)
    source = str(meta.get("source") or "")
    assets = list(meta.get("assets") or [])
    if not assets:
        raise ReturnsError(status_code=500, detail="Run metadata has no assets.")

    sig = _cache_signature(meta, events_path, meta_path)
    cached = _cached_payloads.get(run_id)
    if cached is not None and cached[0] == sig:
        return cached[1]

    if source == "real":
        payload = _build_real_payload(run_id, assets, events_path)
    elif source == "generative":
        events_df = pd.read_parquet(events_path, columns=["t", "asset", "w", "mag"])
        payload = build_generative_payload(run_id=run_id, assets=assets, events_df=events_df)
    else:
        raise ReturnsError(status_code=400, detail="returns panel only available for real/generative runs")

    _cached_payloads[run_id] = (sig, payload)
    return payload
