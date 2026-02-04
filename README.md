---
title: Cascade Flow
emoji: ⚡
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Cascading Extremes Transformer

This repo includes a production-ready **web app** (React/Three + FastAPI) for the immersive cascade viewer, along with the full EVT modeling pipeline.

This repo implements an EVT‑grounded generative model for cascading extremes over crypto returns, using a causal transformer with Hawkes‑style feedback and direction‑dependent thresholds.

## Quickstart

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download data:

```bash
python -m cascades.data.download --config configs/default.yaml
```

3. Train the model:

```bash
python -m cascades.train --config configs/default.yaml
```

4. Simulate cascades:

```bash
python -m cascades.simulate --config configs/default.yaml
```

5. Launch the visualization:

```bash
streamlit run app.py
```

## Immersive Visualizer (Flagship)

1. Export run artifacts:

```bash
python -m cascades.viz_export --config configs/default.yaml --run_id 2026-02-03_real --source real
python -m cascades.viz_export --config configs/default.yaml --run_id 2026-02-03_sim --source simulated
```

2. Start the API:

```bash
cd web/api
uvicorn main:app --reload --port 8000
```

3. Start the immersive frontend:

```bash
cd web/immersive
npm install
npm run dev
```

Or run both with:

```bash
make dev
```

See `docs/immersive_viz.md` for the full architecture and production build steps.

## Frontend (Immersive) — Editing Guide

This frontend is the “cinematic” experience of the paper. Below is a **what‑does‑what** guide so you can edit quickly and confidently.

### Main folder

`web/immersive/`

Key contents:

- `index.html`: loads fonts and defines the `#root` container.
- `vite.config.ts`: Vite configuration (dev port, build).
- `tailwind.config.js`: color palette and typography.
- `src/`: React + Three.js source code.

### Entry points

- `web/immersive/src/main.tsx`
  - React entry point.
  - Mounts the root `App`.

- `web/immersive/src/app.tsx`
  - **Orchestration layer**: loads data from the API, manages the timeline, applies time scaling.
  - Builds metric series and prepares events for 3D rendering.
  - Enables the simulated “cascade window” (asset + magnitude + horizon).

### 3D Scene

- `web/immersive/src/scenes/CascadeScene.tsx`
  - Three.js canvas + camera + lights.
  - Inserts the simplex surface and point cloud.

- `web/immersive/src/scenes/SimplexSurface.tsx`
  - Simplex triangle centered at the origin.
  - Translucent material so it doesn’t hide points.

- `web/immersive/src/scenes/EventsPoints.tsx`
  - Renders points with `PointsMaterial`.
  - Controls visibility via `geometry.setDrawRange()` using `currentTime`.
  - Adjusts size, opacity, z‑fight, and visibility.

### UI / Overlays

- `web/immersive/src/components/Overlay.tsx`
  - Main UI layout above the canvas.
  - Inserts KPIs, sparklines, cascade timeline, and controls.

- `web/immersive/src/components/KPICards.tsx`
  - Live KPIs (time, events, R, λ/ψ, assets).

- `web/immersive/src/components/Sparklines.tsx`
  - Mini time series that **evolve with the playhead**.
  - Lines: λ total, ψ cascade, mean R.

- `web/immersive/src/components/CascadeSignalPanel.tsx`
  - **Cascade probability (ψ/λ)** over time.
  - **Dominant‑asset timeline** with ψ/λ printed on each point.

- `web/immersive/src/components/Controls.tsx`
  - Play/pause, slider, speed.

- `web/immersive/src/components/ScenarioControls.tsx`
  - For simulated runs only.
  - Choose asset + minimum magnitude + window T to “seed” a cascade.

- `web/immersive/src/components/CompareToggle.tsx`
  - Switch between runs (real/simulated).

- `web/immersive/src/components/ErrorBoundary.tsx`
  - Prevents a blank screen if a panel crashes.

### State & utilities

- `web/immersive/src/store/useTimelineStore.ts`
  - Global state: time, play, speed.

- `web/immersive/src/api/client.ts`
  - REST client for `GET /runs`, `/events`, `/metrics`, `/meta`.

- `web/immersive/src/api/types.ts`
  - Zod types validating the data contract.

- `web/immersive/src/utils/color.ts`
  - Viridis palette for point magnitudes.

- `web/immersive/src/utils/time.ts`
  - Time formatting (hours → days).

### Fast editing tips

1. **Change UI styling**: `Overlay.tsx`, `KPICards.tsx`, `Sparklines.tsx`.
2. **Change geometry** (simplex/surface): `SimplexSurface.tsx`.
3. **Change points / animation**: `EventsPoints.tsx`.
4. **Change timeline logic**: `useTimelineStore.ts` + `app.tsx`.

### Quick dev

```bash
cd web/immersive
npm install
npm run dev
```

Open `http://localhost:5173`.

---

## Project Layout

- `cascades/` core model + data pipeline
- `configs/default.yaml` configuration
- `data/raw/` cached prices
- `data/processed/` extracted event sequences
- `artifacts/` trained model + fitted quantile model

## Notes

- Defaults use hourly BTC/ETH/BNB close returns for the last 730 days.
- Extreme events are defined via a direction‑dependent threshold `u_tau(w)` learned by quantile regression.
- The model uses a mixture of Dirichlet for angular direction, truncated Gamma for magnitudes, and exponential inter‑event times driven by Hawkes feedback.
