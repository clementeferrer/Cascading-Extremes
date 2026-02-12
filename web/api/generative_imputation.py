"""Imputation utilities for dense generative return tracks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
PHASE2_DIR = ROOT_DIR / "artifacts" / "phase2"
IMPUTER_PATH = PHASE2_DIR / "imputer_saits.pypots"
IMPUTER_META_PATH = PHASE2_DIR / "imputer_meta.json"
X_TO_RETURNS_MAP_PATH = PHASE2_DIR / "x_to_returns_map.npz"

DEFAULT_ASSETS = ["BTC-USD", "ETH-USD", "BNB-USD"]
EPS = 1.0e-8


class ImputationUnavailable(RuntimeError):
    """Raised when SAITS or supporting artifacts are unavailable."""


@dataclass
class ImputationOutcome:
    matrix: np.ndarray
    mode: str
    fallback_reason: Optional[str]
    anchor_count: int
    horizon_hours: int


def _mtime_ns(path: Path) -> int:
    if not path.exists():
        return -1
    return path.stat().st_mtime_ns


def artifact_signature() -> Tuple[int, int, int]:
    return (_mtime_ns(IMPUTER_PATH), _mtime_ns(IMPUTER_META_PATH), _mtime_ns(X_TO_RETURNS_MAP_PATH))


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


def _event_rows(events_df: pd.DataFrame) -> List[Tuple[float, str, np.ndarray, float]]:
    rows: List[Tuple[float, str, np.ndarray, float]] = []
    for t_val, asset_val, w_val, mag_val in zip(
        events_df["t"].tolist(),
        events_df["asset"].tolist(),
        events_df["w"].tolist(),
        events_df["mag"].tolist(),
    ):
        t_f = float(t_val)
        mag_f = float(mag_val)
        w = np.asarray(w_val, dtype=np.float32).reshape(-1)
        if not np.isfinite(t_f) or not np.isfinite(mag_f) or w.size == 0:
            continue
        rows.append((t_f, str(asset_val), w, mag_f))
    rows.sort(key=lambda item: item[0])
    return rows


def _build_anchor_matrix(
    rows: List[Tuple[float, str, np.ndarray, float]],
    assets: List[str],
    horizon_override: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if not rows:
        return np.zeros((0, len(assets)), dtype=np.float32), np.zeros((0, len(assets)), dtype=bool), 0

    max_t = max(0.0, max(r[0] for r in rows))
    horizon = int(np.ceil(max_t))
    if horizon_override is not None and horizon_override > horizon:
        horizon = int(horizon_override)

    d = len(assets)
    anchors = np.full((horizon + 1, d), np.nan, dtype=np.float32)
    known_mask = np.zeros((horizon + 1, d), dtype=bool)

    for t, _asset, w, mag in rows:
        if t < 0:
            continue
        hour = int(np.clip(np.rint(t), 0, horizon))
        x_vec = w[:d] * mag
        if x_vec.size < d:
            continue
        for j in range(d):
            val = float(x_vec[j])
            if not np.isfinite(val):
                continue
            if not known_mask[hour, j] or abs(val) > abs(float(anchors[hour, j])):
                anchors[hour, j] = val
                known_mask[hour, j] = True

    return anchors, known_mask, horizon


def _load_saits_meta() -> Dict:
    if not IMPUTER_META_PATH.exists():
        raise ImputationUnavailable("missing imputer metadata artifact")
    payload = json.loads(IMPUTER_META_PATH.read_text(encoding="utf-8"))
    if "model" not in payload or not isinstance(payload["model"], dict):
        raise ImputationUnavailable("invalid imputer metadata: model config missing")
    return payload


def _load_saits_model(n_features: int):
    if not IMPUTER_PATH.exists():
        raise ImputationUnavailable("missing SAITS artifact")
    meta = _load_saits_meta()
    model_cfg = dict(meta["model"])
    model_cfg["n_features"] = int(n_features)

    try:
        from pypots.imputation import SAITS  # type: ignore
    except Exception:
        try:
            from pypots.imputation.saits import SAITS  # type: ignore
        except Exception as exc:
            raise ImputationUnavailable("pypots is not installed in runtime") from exc

    try:
        model = SAITS(**model_cfg)
        model.load(str(IMPUTER_PATH))
    except Exception as exc:
        raise ImputationUnavailable(f"failed to load SAITS artifact: {exc}") from exc

    n_steps = int(model_cfg.get("n_steps", 0))
    if n_steps <= 0:
        raise ImputationUnavailable("invalid SAITS configuration: n_steps must be positive")
    return model, n_steps


def _impute_x_with_saits(anchor_x: np.ndarray, known_mask: np.ndarray) -> np.ndarray:
    if anchor_x.size == 0:
        return anchor_x

    n_hours, d = anchor_x.shape
    model, n_steps = _load_saits_model(d)

    stride = max(1, n_steps // 2)
    starts = list(range(0, max(1, n_hours - n_steps + 1), stride))
    tail_start = max(0, n_hours - n_steps)
    if not starts or starts[-1] != tail_start:
        starts.append(tail_start)

    agg = np.zeros((n_hours, d), dtype=np.float64)
    counts = np.zeros((n_hours, d), dtype=np.float64)

    for start in starts:
        end = min(n_hours, start + n_steps)
        window = np.full((n_steps, d), np.nan, dtype=np.float32)
        span = end - start
        window[:span] = anchor_x[start:end]
        try:
            imputed = model.impute({"X": window[None, :, :]})
        except Exception as exc:
            raise ImputationUnavailable(f"SAITS inference failed: {exc}") from exc

        arr = np.asarray(imputed, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.shape != (n_steps, d):
            raise ImputationUnavailable("SAITS output shape mismatch")

        agg[start:end] += arr[:span]
        counts[start:end] += 1.0

    counts = np.maximum(counts, 1.0)
    filled = (agg / counts).astype(np.float32)
    filled[known_mask] = anchor_x[known_mask]

    for j in range(d):
        col = filled[:, j]
        mask = np.isfinite(col)
        if not mask.any():
            filled[:, j] = 0.0
            continue
        idx = np.arange(n_hours)
        filled[:, j] = np.interp(idx, idx[mask], col[mask]).astype(np.float32)

    return filled


def _load_x_to_returns_map(assets: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if not X_TO_RETURNS_MAP_PATH.exists():
        raise ImputationUnavailable("missing x->returns mapping artifact")
    try:
        payload = np.load(str(X_TO_RETURNS_MAP_PATH), allow_pickle=False)
    except Exception as exc:
        raise ImputationUnavailable(f"failed to read mapping artifact: {exc}") from exc

    if "assets" not in payload or "x_sorted" not in payload or "ret_sorted" not in payload:
        raise ImputationUnavailable("invalid mapping artifact schema")

    map_assets = [str(a) for a in payload["assets"].tolist()]
    x_sorted = np.asarray(payload["x_sorted"], dtype=np.float64)
    ret_sorted = np.asarray(payload["ret_sorted"], dtype=np.float64)
    if x_sorted.ndim != 2 or ret_sorted.ndim != 2 or x_sorted.shape != ret_sorted.shape:
        raise ImputationUnavailable("invalid mapping artifact array shapes")
    if x_sorted.shape[0] != len(map_assets):
        raise ImputationUnavailable("mapping asset list does not match mapping matrix shape")

    wanted = [_normalize_asset_name(a) for a in assets]
    available = {_normalize_asset_name(a): i for i, a in enumerate(map_assets)}
    order = []
    for name in wanted:
        if name in available:
            order.append(available[name])
            continue
        if name.endswith("-USD") and name[:-4] in available:
            order.append(available[name[:-4]])
            continue
        with_usd = f"{name}-USD"
        if with_usd in available:
            order.append(available[with_usd])
            continue
        raise ImputationUnavailable(f"mapping missing asset: {name}")

    return x_sorted[order], ret_sorted[order], map_assets


def _map_x_to_returns_pct(x_filled: np.ndarray, assets: List[str]) -> np.ndarray:
    if x_filled.size == 0:
        return x_filled
    x_sorted, ret_sorted, _ = _load_x_to_returns_map(assets)
    d = x_filled.shape[1]
    if x_sorted.shape[0] != d:
        raise ImputationUnavailable("mapping dimension mismatch")

    out = np.zeros_like(x_filled, dtype=np.float32)
    for j in range(d):
        xs = np.asarray(x_sorted[j], dtype=np.float64)
        rs = np.asarray(ret_sorted[j], dtype=np.float64)
        xs = xs[np.isfinite(xs)]
        rs = rs[np.isfinite(rs)]
        n = min(len(xs), len(rs))
        if n < 2:
            raise ImputationUnavailable("mapping has insufficient support points")
        xs = np.sort(xs[:n])
        rs = np.sort(rs[:n])

        vals = x_filled[:, j].astype(np.float64)
        ranks = np.searchsorted(xs, vals, side="right")
        u = ranks / float(len(xs))
        q = np.linspace(0.0, 1.0, len(rs), dtype=np.float64)
        mapped = np.interp(u, q, rs)
        out[:, j] = mapped.astype(np.float32)
    return out


def _build_event_only_payload(
    run_id: str,
    assets: List[str],
    rows: List[Tuple[float, str, np.ndarray, float]],
    reason: str,
) -> Dict:
    d = len(assets)
    asset_lookup = _asset_lookup(assets)

    series: Dict[str, List[List[float]]] = {a: [] for a in assets}
    extreme_points: Dict[str, List[List[float]]] = {a: [] for a in assets}
    for t, event_asset, w, mag in rows:
        x_vec = w[:d] * mag
        for j, asset in enumerate(assets):
            val = float(x_vec[j]) if j < x_vec.size else np.nan
            if np.isfinite(val):
                series[asset].append([float(t), val])

        resolved = _resolve_event_asset(event_asset, asset_lookup)
        if resolved is None:
            continue
        idx = assets.index(resolved)
        if idx < x_vec.size and np.isfinite(x_vec[idx]):
            extreme_points[resolved].append([float(t), float(x_vec[idx])])

    return {
        "run_id": run_id,
        "units": "log_return_pct",
        "assets": assets,
        "series": series,
        "extreme_points": extreme_points,
        "alignment": {
            "method": "event_only_projection_rw",
            "offset_hours": 0,
            "candidate_count": int(len(rows)),
            "start_datetime_utc": None,
        },
        "count": int(len(rows)),
        "series_mode": "generative_event_only_fallback",
        "imputation": {
            "method": "saits",
            "anchor_method": "nearest_hour_max_abs",
            "anchor_count": int(len(rows)),
            "horizon_hours": int(np.ceil(rows[-1][0])) if rows else 0,
            "fallback_reason": reason,
        },
    }


def build_generative_payload(
    run_id: str,
    assets: List[str],
    events_df: pd.DataFrame,
    horizon_hours: Optional[int] = None,
) -> Dict:
    assets = list(assets or DEFAULT_ASSETS)
    rows = _event_rows(events_df)
    if not rows:
        return {
            "run_id": run_id,
            "units": "log_return_pct",
            "assets": assets,
            "series": {a: [] for a in assets},
            "extreme_points": {a: [] for a in assets},
            "alignment": {
                "method": "saits_hourly_imputation",
                "offset_hours": 0,
                "candidate_count": 0,
                "start_datetime_utc": None,
            },
            "count": 0,
            "series_mode": "generative_event_only_fallback",
            "imputation": {
                "method": "saits",
                "anchor_method": "nearest_hour_max_abs",
                "anchor_count": 0,
                "horizon_hours": int(horizon_hours or 0),
                "fallback_reason": "no events available",
            },
        }

    anchor_x, known_mask, horizon = _build_anchor_matrix(rows, assets, horizon_override=horizon_hours)
    anchor_count = int(np.count_nonzero(known_mask))
    try:
        x_filled = _impute_x_with_saits(anchor_x, known_mask)
        returns_pct = _map_x_to_returns_pct(x_filled, assets)
    except ImputationUnavailable as exc:
        return _build_event_only_payload(run_id, assets, rows, reason=str(exc))

    d = len(assets)
    series: Dict[str, List[List[float]]] = {}
    for j, asset in enumerate(assets):
        col = returns_pct[:, j]
        series[asset] = [[float(t), float(v)] for t, v in enumerate(col) if np.isfinite(v)]

    asset_lookup = _asset_lookup(assets)
    extreme_points: Dict[str, List[List[float]]] = {a: [] for a in assets}
    for t, event_asset, _w, _mag in rows:
        resolved = _resolve_event_asset(event_asset, asset_lookup)
        if resolved is None:
            continue
        hour = int(np.clip(np.rint(t), 0, horizon))
        idx = assets.index(resolved)
        val = float(returns_pct[hour, idx])
        if np.isfinite(val):
            extreme_points[resolved].append([float(hour), val])

    return {
        "run_id": run_id,
        "units": "log_return_pct",
        "assets": assets,
        "series": series,
        "extreme_points": extreme_points,
        "alignment": {
            "method": "saits_hourly_imputation",
            "offset_hours": 0,
            "candidate_count": int(anchor_count),
            "start_datetime_utc": None,
        },
        "count": int(len(rows)),
        "series_mode": "generative_imputed",
        "imputation": {
            "method": "saits",
            "anchor_method": "nearest_hour_max_abs",
            "anchor_count": int(anchor_count),
            "horizon_hours": int(horizon),
        },
    }
