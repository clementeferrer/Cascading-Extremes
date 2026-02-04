from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
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
    from cascades.simulate import load_model, load_quantile_model, generate_with_limits
    from cascades.extremes import QuantileModelConfig
    from cascades.utils import load_config
    from cascades.viz_export.export import export_run_from_arrays

    HAS_CASCADES = True
except ModuleNotFoundError:
    HAS_CASCADES = False


ROOT_DIR = Path(__file__).resolve().parents[2]


def _dist_dir() -> Path:
    static_env = os.getenv("IMMERSIVE_STATIC")
    return Path(static_env) if static_env else ROOT_DIR / "web" / "immersive" / "dist"



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


class GenerateRequest(BaseModel):
    seed_run_id: Optional[str] = None
    seed_asset: Optional[str] = None
    min_mag: Optional[float] = None
    seed_window: int = 64
    max_events: int = 256
    max_time: Optional[float] = 240.0
    temperature: Optional[float] = 1.0
    top_k: Optional[int] = 0
    config: str = "configs/default.yaml"
    seed: Optional[int] = None  # Random seed for reproducibility


@app.post("/generate/continue")
def generate_continue(req: GenerateRequest):
    if not HAS_CASCADES:
        raise HTTPException(
            status_code=500,
            detail="Generation requires cascades package. Run the API from the repo root.",
        )
    # Initialize random seed for reproducibility
    if req.seed is not None:
        random.seed(req.seed)
        np.random.seed(req.seed)
    runs = list_runs()
    seed_run_id = req.seed_run_id
    if not seed_run_id:
        # Prefer real runs, fallback to first available
        real_runs = [r for r in runs if r.get("source") == "real"]
        seed_run_id = real_runs[0]["run_id"] if real_runs else (runs[0]["run_id"] if runs else None)
    if not seed_run_id:
        raise HTTPException(status_code=400, detail="No seed runs available")

    events = read_events(seed_run_id, 0, 200000)
    if not events:
        raise HTTPException(status_code=404, detail="Seed run has no events")

    W = np.array([e["w"] for e in events], dtype=np.float32)
    R = np.array([e["mag"] for e in events], dtype=np.float32)
    T = np.array([e["t"] for e in events], dtype=np.float32)
    if len(T) < 2:
        dT = np.ones_like(T, dtype=np.float32)
    else:
        dT = np.diff(T, prepend=T[0]).astype(np.float32)
        dT[0] = np.median(dT[1:]) if len(dT) > 1 else 1.0

    seed_window = max(4, min(req.seed_window, len(T)))
    mask = np.ones(len(T), dtype=bool)
    if req.seed_asset:
        mask &= np.array([events[i]["asset"] == req.seed_asset for i in range(len(events))], dtype=bool)
    if req.min_mag is not None:
        mask &= np.array([events[i]["mag"] >= req.min_mag for i in range(len(events))], dtype=bool)
    valid_idx = np.where(mask)[0]
    if len(valid_idx) == 0:
        valid_idx = np.arange(len(T))
    trigger_idx = int(valid_idx[-1])
    start = max(0, trigger_idx - seed_window + 1)
    seed = {
        "T": T[start : trigger_idx + 1],
        "dT": dT[start : trigger_idx + 1],
        "W": W[start : trigger_idx + 1],
        "R": R[start : trigger_idx + 1],
    }

    model_path = Path("artifacts") / "model.pt"
    q_path = Path("artifacts") / "quantile_model.pt"
    max_time = req.max_time if req.max_time is not None else 240.0

    trim_start = 0
    if model_path.exists() and q_path.exists():
        cfg = load_config(req.config)
        model = load_model(str(model_path))
        q_cfg = QuantileModelConfig(**cfg["extremes"]["quantile_model"])
        q_model = load_quantile_model(str(q_path), model.d_assets, q_cfg)
        sim = generate_with_limits(seed, req.max_events, max_time, model, q_model)
        trim_start = max(0, len(seed["T"]) - 1)
    else:
        # Fallback: random trigger segment after trigger within horizon
        trigger_idx = int(random.choice(valid_idx))
        t0 = T[trigger_idx]
        mask_time = (T >= t0) & ((T - t0) <= max_time)
        idx = np.where(mask_time)[0][: req.max_events]
        if len(idx) == 0:
            idx = np.array([trigger_idx])
        sim = {
            "T": T[idx],
            "dT": dT[idx],
            "W": W[idx],
            "R": R[idx],
        }
        trim_start = 0

    # Keep only trigger + generated continuation, then rebase time to start at 0
    for key in ("T", "dT", "W", "R"):
        sim[key] = sim[key][trim_start:]
    sim["T"] = sim["T"] - sim["T"][0]
    if len(sim["T"]) > 1:
        sim["dT"] = np.diff(sim["T"], prepend=sim["T"][0]).astype(np.float32)
        sim["dT"][0] = np.median(sim["dT"][1:]) if len(sim["dT"]) > 1 else 1.0

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ_gen")
    export_run_from_arrays(req.config, run_id, "generative", sim)
    return {"run_id": run_id}


@app.post("/generate")
def generate(req: GenerateRequest):
    return generate_continue(req)


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
