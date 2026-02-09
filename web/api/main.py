from __future__ import annotations

import json
import os
import random
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
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


def _resolve_config_path(config_path: str) -> Path:
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = ROOT_DIR / cfg_path
    return cfg_path


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
    theta: float = 0.0            # Azimuthal angle theta in [0, 2*pi]
    phi: float = 1.5708           # Polar angle phi in [0, pi], default pi/2 (BTC axis)
    magnitude: float = 3.0        # Radial magnitude R of the initial shock
    max_time: float = 240.0       # Time horizon (hours) — the only stopping criterion
    config: str = "configs/phase2.yaml"
    seed: Optional[int] = None    # Random seed for reproducibility


@app.post("/generate/continue")
def generate_continue(req: GenerateRequest):
    if not HAS_PHASE2:
        raise HTTPException(
            status_code=500,
            detail="Phase 2 model required for generation.",
        )

    # Load Phase 2 model
    p2_model_path = ARTIFACTS_DIR / "phase2" / "model.pt"
    p2_q_path = ARTIFACTS_DIR / "phase2" / "quantile_model.pt"
    if not p2_model_path.exists() or not p2_q_path.exists():
        raise HTTPException(status_code=500, detail="Phase 2 model artifacts not found.")

    if req.seed is not None:
        random.seed(req.seed)
        np.random.seed(req.seed)

    cfg_path = _resolve_config_path("configs/phase2.yaml")
    cfg = load_config(str(cfg_path))
    model = load_model_p2(str(p2_model_path))
    model.to("cpu")
    model.eval()
    q_cfg = QCfgP2(**cfg["extremes"]["quantile_model"])
    q_model = load_qmodel_p2(str(p2_q_path), model.d_assets, q_cfg)
    q_model.to("cpu")
    q_model.eval()

    # Convert spherical coordinates to direction on S^2
    theta, phi = req.theta, req.phi
    w0 = np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ], dtype=np.float32)
    # Normalize (safety)
    norm = np.linalg.norm(w0)
    if norm > 1e-8:
        w0 = w0 / norm

    # Initial magnitude: user-provided R, clamped above threshold
    u0 = q_model(torch.tensor(w0[None, :], dtype=torch.float32)).item()
    r0 = max(req.magnitude, u0 + 0.01)

    # Autoregressive generation — only max_time stops the cascade
    sim = autoregressive_generate(w0, r0, req.max_time, model, q_model)

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ_gen")
    cfg_path = _resolve_config_path(req.config)
    export_run_from_arrays(str(cfg_path), run_id, "generative", sim, output_dir=str(RUNS_DIR))
    return {"run_id": run_id}


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
