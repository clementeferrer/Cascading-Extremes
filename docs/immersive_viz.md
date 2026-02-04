# Immersive Cascading Extremes Visualizer

## Overview
The immersive visualizer is a production-grade WebGL experience for EVT cascades. It reads standardized run artifacts (meta + events + metrics) and renders a cinematic, synchronized 3D scene with minimal overlays. The Streamlit app (`app.py`) remains the lightweight viewer for quick checks.

## Architecture
- `cascades/viz_export`: exports run artifacts into a stable schema.
- `web/api`: FastAPI server serving run metadata and event chunks.
- `web/immersive`: React + Three.js WebGL client with timeline playback.

## Run Artifacts
Each run lives under `artifacts/runs/<run_id>/`:
- `meta.json`: run metadata (assets, threshold, checkpoint, config hash).
- `events.parquet`: event records (time, direction, magnitude, etc.).
- `metrics.parquet`: per-event metrics (lambda, psi, mean magnitude, counts).
- `artifacts/runs/index.json`: registry for all runs.

## Export Pipeline
Export real or simulated runs using the CLI:

```bash
python -m cascades.viz_export --config configs/default.yaml --run_id 2026-02-03_real --source real
python -m cascades.viz_export --config configs/default.yaml --run_id 2026-02-03_sim --source simulated
```

## API Server
Run the API from `web/api`:

```bash
cd web/api
uvicorn main:app --reload --port 8000
```

The API will serve:
- `GET /runs`
- `GET /runs/{run_id}/meta`
- `GET /runs/{run_id}/events?offset=0&limit=5000`
- `GET /runs/{run_id}/metrics?offset=0&limit=5000`
- `GET /runs/{run_id}/summary`
- `GET /runs/{run_id}/download?file=events|metrics`

## Immersive Frontend
Run the WebGL client:

```bash
cd web/immersive
npm install
npm run dev
```

By default the client talks to `http://localhost:8000`. Override with:

```bash
VITE_API_URL=http://localhost:8000 npm run dev
```

## Dev Workflow
From repo root:

```bash
make dev
```

This starts the API and the frontend in the same terminal. Use `CTRL+C` to stop both.

## Production Build
Build the frontend:

```bash
make build
```

Serve API and static assets together:

```bash
make serve
```

The immersive app will be served at `http://localhost:8000/`.

## Notes
- If the API cannot find `artifacts/runs`, verify you exported runs from the repo root.
- Metrics like lambda/psi are optional when no checkpoint is available.
- For large runs, the API returns chunked pages; the client can request larger limits as needed.
