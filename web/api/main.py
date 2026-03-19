from __future__ import annotations

import json
import os
import random
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from web.api.metrics import summarize
    from web.api.storage import get_run_path, list_runs, read_events, read_meta, read_metrics
except ModuleNotFoundError:
    # Fallback when running from web/api directly (e.g. `uvicorn main:app`)
    from metrics import summarize  # type: ignore
    from storage import get_run_path, list_runs, read_events, read_meta, read_metrics  # type: ignore

try:
    from cascades.utils import load_config
    from cascades.viz_export.export import export_run_from_arrays

    HAS_CASCADES = True
except ModuleNotFoundError:
    HAS_CASCADES = False

try:
    from web.api.bulk import get_bulk_positions
    HAS_BULK = True
except ModuleNotFoundError:
    try:
        from bulk import get_bulk_positions  # type: ignore
        HAS_BULK = True
    except ModuleNotFoundError:
        HAS_BULK = False

try:
    from web.api.returns import ReturnsError, get_returns_payload
    HAS_RETURNS = True
except ModuleNotFoundError:
    try:
        from returns import ReturnsError, get_returns_payload  # type: ignore
        HAS_RETURNS = True
    except ModuleNotFoundError:
        ReturnsError = Exception  # type: ignore[assignment]
        HAS_RETURNS = False

# Phase 2 generation (vMF + Ogata thinning on the sphere)
try:
    from second_phase.simulate import load_model as load_model_p2, load_quantile_model as load_qmodel_p2, autoregressive_generate
    from second_phase.extremes import QuantileModelConfig as QCfgP2

    HAS_PHASE2 = True
except ModuleNotFoundError:
    HAS_PHASE2 = False


logger = logging.getLogger("cascade")
logging.basicConfig(level=logging.INFO)

ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
CONFIGS_DIR = ROOT_DIR / "configs"
RUNS_DIR = ARTIFACTS_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
PHASE2_META_PATH = ARTIFACTS_DIR / "phase2" / "meta.json"
PRICES_PATH = ROOT_DIR / "data" / "raw" / "prices_1h_730d.csv"

# ---------------------------------------------------------------------------
# Cached Phase 2 model singletons (avoid reloading on every request)
# ---------------------------------------------------------------------------
_P2_MODEL = None
_P2_QMODEL = None
_P2_CFG = None
_P2_CFG_SYMBOLS: list[str] = []
_P2_MODEL_SYMBOLS: list[str] = []


def _get_phase2():
    """Return (model, q_model, cfg, cfg_symbols, model_symbols), loading once."""
    global _P2_MODEL, _P2_QMODEL, _P2_CFG, _P2_CFG_SYMBOLS, _P2_MODEL_SYMBOLS
    if _P2_MODEL is not None:
        return _P2_MODEL, _P2_QMODEL, _P2_CFG, _P2_CFG_SYMBOLS, _P2_MODEL_SYMBOLS

    p2_model_path = ARTIFACTS_DIR / "phase2" / "model.pt"
    p2_q_path = ARTIFACTS_DIR / "phase2" / "quantile_model.pt"
    if not p2_model_path.exists() or not p2_q_path.exists():
        raise HTTPException(status_code=500, detail="Phase 2 model artifacts not found.")

    cfg_path = _resolve_config_path("configs/phase2.yaml")
    cfg = load_config(str(cfg_path))
    cfg_symbols = [str(s) for s in cfg.get("data", {}).get("symbols", [])]
    model_symbols = _load_phase2_symbols(default_symbols=cfg_symbols)

    model = load_model_p2(str(p2_model_path))
    model.to("cpu")
    model.eval()
    q_cfg = QCfgP2(**cfg["extremes"]["quantile_model"])
    q_model = load_qmodel_p2(str(p2_q_path), model.d_assets, q_cfg)
    q_model.to("cpu")
    q_model.eval()

    _P2_MODEL = model
    _P2_QMODEL = q_model
    _P2_CFG = cfg
    _P2_CFG_SYMBOLS = cfg_symbols
    _P2_MODEL_SYMBOLS = model_symbols
    logger.info("Phase 2 models loaded and cached.")
    return model, q_model, cfg, cfg_symbols, model_symbols


def _resolve_config_path(config_path: str) -> Path:
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = ROOT_DIR / cfg_path
    return cfg_path


def _dist_dir() -> Path:
    static_env = os.getenv("IMMERSIVE_STATIC")
    return Path(static_env) if static_env else ROOT_DIR / "web" / "immersive" / "dist"


def _max_horizon_hours(default: int = 17468) -> int:
    if PRICES_PATH.exists():
        try:
            n = len(pd.read_csv(PRICES_PATH, usecols=[0]))
            if n >= 2:
                return int(n - 1)
        except Exception:
            pass
    return int(default)


def _laplace_quantile(u: np.ndarray) -> np.ndarray:
    """Inverse CDF of standard Laplace used by return-to-margin mapping."""
    centered = u - 0.5
    return -np.sign(centered) * np.log(np.maximum(1.0 - 2.0 * np.abs(centered), 1e-15))


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace("_", "-")


def _load_phase2_symbols(default_symbols: Optional[list[str]] = None) -> list[str]:
    if PHASE2_META_PATH.exists():
        try:
            payload = json.loads(PHASE2_META_PATH.read_text(encoding="utf-8"))
            symbols = payload.get("symbols")
            if isinstance(symbols, list) and all(isinstance(s, str) for s in symbols):
                return [str(s) for s in symbols]
        except Exception:
            pass
    return list(default_symbols or [])


def _build_permutation(src_symbols: list[str], dst_symbols: list[str]) -> Optional[list[int]]:
    src_norm = [_normalize_symbol(s) for s in src_symbols]
    dst_norm = [_normalize_symbol(s) for s in dst_symbols]
    idx: list[int] = []
    for symbol in dst_norm:
        if symbol in src_norm:
            idx.append(src_norm.index(symbol))
            continue
        if symbol.endswith("-USD") and symbol[:-4] in src_norm:
            idx.append(src_norm.index(symbol[:-4]))
            continue
        with_usd = f"{symbol}-USD"
        if with_usd in src_norm:
            idx.append(src_norm.index(with_usd))
            continue
        return None
    return idx


def _reorder_w_columns(W: np.ndarray, src_symbols: list[str], dst_symbols: list[str]) -> np.ndarray:
    if W.ndim != 2 or not src_symbols or not dst_symbols:
        return W
    perm = _build_permutation(src_symbols, dst_symbols)
    if perm is None or len(perm) != W.shape[1]:
        return W
    return W[:, perm]


def _reorder_vector(vec: np.ndarray, src_symbols: list[str], dst_symbols: list[str]) -> np.ndarray:
    if vec.ndim != 1 or not src_symbols or not dst_symbols:
        return vec
    perm = _build_permutation(src_symbols, dst_symbols)
    if perm is None or len(perm) != vec.shape[0]:
        return vec
    return vec[perm]


def _latest_real_run_id() -> Optional[str]:
    for run in reversed(list_runs()):
        if run.get("source") == "real":
            run_id = run.get("run_id")
            if isinstance(run_id, str) and run_id:
                return run_id
    return None


def _load_real_context_prompt(context_run_id: Optional[str]) -> dict[str, np.ndarray]:
    preferred = context_run_id or "real_latest"
    run_id = preferred
    try:
        meta = read_meta(run_id)
    except Exception:
        fallback = _latest_real_run_id()
        if not fallback:
            raise HTTPException(status_code=500, detail="No real run available for completion context.")
        run_id = fallback
        meta = read_meta(run_id)

    if meta.get("source") != "real":
        raise HTTPException(status_code=400, detail="context_run_id must reference a real run.")

    events_path = get_run_path(run_id) / "events.parquet"
    if not events_path.exists():
        raise HTTPException(status_code=500, detail=f"Context events not found for run: {run_id}")

    df = pd.read_parquet(events_path)
    if df.empty:
        raise HTTPException(status_code=500, detail=f"Context run has no events: {run_id}")

    if "t" not in df.columns or "w" not in df.columns or "mag" not in df.columns:
        raise HTTPException(status_code=500, detail="Context run is missing required event columns.")

    df = df.sort_values("t")
    W = np.vstack(df["w"].to_numpy()).astype(np.float32)
    R = df["mag"].to_numpy(np.float32)
    T_raw = df["t"].to_numpy(np.float32)

    if len(T_raw) < 1:
        raise HTTPException(status_code=500, detail="Context run has invalid timeline.")

    # Shift timeline so the latest real event sits at t=0.
    T = T_raw - float(T_raw[-1])
    dT = np.diff(T, prepend=T[0]).astype(np.float32)
    if len(dT) > 1:
        med = float(np.median(dT[1:]))
        dT[0] = med if np.isfinite(med) and med > 1.0e-6 else 1.0
    elif len(dT) == 1:
        dT[0] = 1.0

    return {"W": W, "R": R, "T": T.astype(np.float32), "dT": dT.astype(np.float32)}



app = FastAPI(title="Cascading Extremes Immersive API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/runs")
def runs():
    return {"runs": list_runs()}


@app.get("/runs/{run_id}/meta")
def run_meta(run_id: str):
    try:
        return read_meta(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc


@app.get("/runs/{run_id}/events")
def run_events(
    run_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(5000, ge=1, le=200000),
):
    path = get_run_path(run_id) / "events.parquet"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Events not found")
    records = read_events(run_id, offset, limit)
    return {"events": records, "offset": offset, "limit": limit, "count": len(records)}


@app.get("/runs/{run_id}/metrics")
def run_metrics(
    run_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(5000, ge=1, le=200000),
):
    path = get_run_path(run_id) / "metrics.parquet"
    if not path.exists():
        return {"metrics": [], "offset": offset, "limit": limit, "count": 0}
    records = read_metrics(run_id, offset, limit)

    # parse per_asset_counts if it is JSON string
    for r in records:
        pac = r.get("per_asset_counts")
        if isinstance(pac, str):
            try:
                r["per_asset_counts"] = json.loads(pac)
            except Exception:
                pass
    return {"metrics": records, "offset": offset, "limit": limit, "count": len(records)}


@app.get("/runs/{run_id}/returns")
def run_returns(run_id: str):
    if not HAS_RETURNS:
        raise HTTPException(status_code=500, detail="Returns computation not available.")
    try:
        return get_returns_payload(run_id)
    except ReturnsError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    except Exception as exc:
        logger.exception("GET /runs/%s/returns failed", run_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/runs/{run_id}/summary")
def run_summary(run_id: str):
    events = read_events(run_id, 0, 100000)
    metrics = read_metrics(run_id, 0, 100000)
    return summarize(events, metrics)


@app.get("/runs/{run_id}/download")
def run_download(run_id: str, file: str):
    if file not in {"events", "metrics"}:
        raise HTTPException(status_code=400, detail="Invalid file")
    path = get_run_path(run_id) / f"{file}.parquet"
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


@app.get("/bulk")
def bulk_observations():
    if not HAS_BULK:
        raise HTTPException(status_code=500, detail="Bulk computation not available.")
    try:
        positions = get_bulk_positions()
        return {"points": positions.tolist(), "count": len(positions)}
    except Exception as exc:
        logger.exception("GET /bulk failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class GenerateRequest(BaseModel):
    theta: float = 0.0            # Azimuthal angle theta in [0, 2*pi]
    phi: float = 1.5708           # Polar angle phi in [0, pi], default pi/2 (BTC axis)
    magnitude: float = 3.0        # Radial magnitude R of the initial shock
    max_time: float = 240.0       # Time horizon (hours) — the only stopping criterion
    temperature: float = 1.0      # Decoding temperature for sampling
    config: str = "configs/phase2.yaml"
    seed: Optional[int] = None    # Random seed for reproducibility


@app.post("/generate/continue")
def generate_continue(req: GenerateRequest):
    if not HAS_PHASE2:
        raise HTTPException(
            status_code=500,
            detail="Phase 2 model required for generation.",
        )

    model, q_model, cfg, cfg_symbols, model_symbols = _get_phase2()

    if req.seed is not None:
        random.seed(req.seed)
        np.random.seed(req.seed)

    # Convert spherical coordinates to direction on S^2
    theta, phi = req.theta, req.phi
    w0 = np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ], dtype=np.float32)
    # User-facing spherical controls follow cfg asset ordering (typically BTC, ETH, BNB).
    # Convert to model internal ordering when they differ.
    w0 = _reorder_vector(w0, cfg_symbols, model_symbols).astype(np.float32)
    # Normalize (safety)
    norm = np.linalg.norm(w0)
    if norm > 1e-8:
        w0 = w0 / norm

    # Initial magnitude: user-provided R, clamped above threshold
    u0 = q_model(torch.tensor(w0[None, :], dtype=torch.float32)).item()
    r0 = max(req.magnitude, u0 + 0.01)

    # Autoregressive generation — only max_time stops the cascade
    max_horizon = _max_horizon_hours()
    max_time = float(np.clip(req.max_time, 1.0, float(max_horizon)))
    sim = autoregressive_generate(w0, r0, max_time, model, q_model, temperature=req.temperature)

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ_gen")
    export_cfg_path = _resolve_config_path(req.config)
    export_run_from_arrays(str(export_cfg_path), run_id, "generative", sim, output_dir=str(RUNS_DIR))
    return {"run_id": run_id}


class GenerateFromReturnsRequest(BaseModel):
    returns: dict[str, float]  # e.g. {"BTC-USD": -5.0, "ETH-USD": -3.0, "BNB-USD": -1.0}
    max_time: float = 240.0
    temperature: float = 1.0
    context_run_id: Optional[str] = None
    seed: Optional[int] = None


@app.post("/generate/from-returns")
def generate_from_returns(req: GenerateFromReturnsRequest):
    if not HAS_PHASE2:
        raise HTTPException(status_code=500, detail="Phase 2 model required for generation.")

    cdfs_path = ARTIFACTS_DIR / "phase2" / "cdfs.npz"
    if not cdfs_path.exists():
        raise HTTPException(status_code=500, detail="Phase 2 CDF artifacts not found.")

    try:
        from cascades.utils import EmpiricalCDF

        model, q_model, cfg, cfg_symbols, model_symbols = _get_phase2()

        if req.seed is not None:
            random.seed(req.seed)
            np.random.seed(req.seed)

        # Load CDFs
        cdf_data = np.load(str(cdfs_path))
        # CDF file order matches the model's training column order.
        asset_order = [str(a) for a in cdf_data.files]

        # Convert % returns to Laplace margins
        X_vals = []
        for asset in asset_order:
            if asset not in req.returns:
                raise HTTPException(status_code=400, detail=f"Missing return input for asset: {asset}")
            sorted_vals = cdf_data[asset]
            cdf = EmpiricalCDF(sorted_values=sorted_vals, eps=1e-6)
            r_decimal = req.returns[asset] / 100.0
            u = cdf.cdf(np.array([r_decimal]))[0]
            x = _laplace_quantile(np.array([u]))[0]
            X_vals.append(x)

        X = np.array(X_vals, dtype=np.float32)
        # Normalize by log(n/2)
        n = max(len(cdf_data[asset_order[0]]), 2)
        X = X / np.log(n / 2)

        # Compute R, W
        R = float(np.linalg.norm(X))
        if R < 1e-8:
            return {"extreme": False, "R": R, "threshold": 0.0,
                    "message": "Returns too close to zero to form a direction."}
        W = X / R

        # Check threshold
        u_tau = q_model(torch.tensor(W[None, :], dtype=torch.float32)).item()

        if R <= u_tau:
            return {
                "extreme": False,
                "R": round(R, 4),
                "threshold": round(u_tau, 4),
                "message": f"Not extreme: R={R:.3f} <= threshold={u_tau:.3f}. Try larger returns.",
            }

        # Completion from full real context -> append seed (W,R) -> autoregressive continuation.
        prompt = _load_real_context_prompt(req.context_run_id)
        max_horizon = _max_horizon_hours(default=n - 1)
        max_time = float(np.clip(req.max_time, 1.0, float(max_horizon)))
        sim = autoregressive_generate(
            W,
            R,
            max_time,
            model,
            q_model,
            temperature=req.temperature,
            prompt=prompt,
        )
        # Keep internal simulation in model order, but export in cfg/UI order
        # so labels and controls remain aligned (BTC/ETH/BNB).
        sim_export = dict(sim)
        sim_export["W"] = _reorder_w_columns(np.asarray(sim["W"], dtype=np.float32), model_symbols, cfg_symbols)
        run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ_gen")
        export_cfg_path = _resolve_config_path("configs/phase2.yaml")
        export_run_from_arrays(str(export_cfg_path), run_id, "generative", sim_export, output_dir=str(RUNS_DIR))
        return {"extreme": True, "run_id": run_id}

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("POST /generate/from-returns failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/generate")
def generate(req: GenerateRequest):
    try:
        return generate_continue(req)
    except Exception as exc:
        logger.exception("POST /generate failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _maybe_mount_static() -> None:
    dist_dir = _dist_dir()
    assets_dir = dist_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")


_maybe_mount_static()


@app.get("/", include_in_schema=False)
def spa_root():
    dist_dir = _dist_dir()
    index_path = dist_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not built")
    return FileResponse(index_path)


@app.get("/{full_path:path}", include_in_schema=False)
def spa_fallback(full_path: str):
    if full_path.startswith(("api", "runs", "docs", "redoc", "openapi.json", "assets")):
        raise HTTPException(status_code=404)
    dist_dir = _dist_dir()
    candidate = dist_dir / full_path
    if candidate.is_file():
        return FileResponse(candidate)
    index_path = dist_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not built")
    return FileResponse(index_path)
