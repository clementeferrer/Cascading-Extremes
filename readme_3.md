# Cascading Extremes — Immersive Frontend Deep Dive

This document is the **frontend specification** for the immersive web experience, with a focus on the **“Launch Immersive Viewer”** flow. It is written for researchers, designers, and engineers who want to understand or modify the landing page and the 3D viewer.

---

## 1) Purpose and Audience

- This document explains the **webpage structure, routing, UI modules, and 3D scene** for the immersive viewer.
- It complements the math in `readme_2.md` and the system overview in `docs/immersive_viz.md`.
- You should use it when editing the **landing page** or **viewer experience**.

---

## 2) Routes and Navigation

- `/` is the **landing page** with narrative and CTA.
- `/viewer` is the **immersive WebGL viewer**.
- The **“Launch Immersive Viewer”** CTA always links to `/viewer`.

Routing is defined in `web/immersive/src/app.tsx`.

---

## 3) Landing Page (Launch Immersive Viewer Entry)

**File:** `web/immersive/src/pages/LandingPage.tsx`

The landing page is cinematic and narrative. It establishes the scientific story and funnels the user to the viewer.

**Primary modules (in order):**
- Top navigation: `web/immersive/src/components/landing/TopNav.tsx`
- Hero headline + CTA: `web/immersive/src/components/landing/Hero.tsx`
- Story sections: `web/immersive/src/components/landing/StorySection.tsx`
- Pipeline steps: `web/immersive/src/components/landing/PipelineSteps.tsx`
- Counters: `web/immersive/src/components/landing/Counters.tsx`
- Logo cloud: `web/immersive/src/components/landing/LogoCloud.tsx`
- Accordion cards: `web/immersive/src/components/landing/AccordionCards.tsx`
- Stepper: `web/immersive/src/components/landing/Stepper.tsx`
- Footer CTA: `web/immersive/src/components/landing/FooterCTA.tsx`

**Visual language**
- Dark, cinematic background.
- Minimal typography with generous spacing.
- Narrative blocks that read like a live research story.

---

## 4) Viewer Page Layout (Core of the Experience)

**File:** `web/immersive/src/pages/ViewerPage.tsx`

The viewer is a **3‑column grid** that avoids any overlap with the 3D canvas.

**Layout regions:**
- **Left column:** controls, KPIs, signal overview, generation controls.
- **Center:** 3D cube scene (unobstructed).
- **Right column:** event rail (minimal timeline).

This layout ensures the cube is always readable, and no UI panel covers the 3D scene.

---

## 5) 3D Scene (Cube + Simplex)

**Files:**
- `web/immersive/src/scenes/CascadeScene.tsx`
- `web/immersive/src/scenes/SimplexPlane.tsx`
- `web/immersive/src/scenes/EventsPoints.tsx`
- `web/immersive/src/utils/geometry.ts`

**Key ideas:**
- The **unit cube** is the primary geometric reference.
- The **simplex plane** is a **visible overlay** inside the cube.
- The **point cloud** represents cascade events.

**Controls:**
- Orbit rotation is enabled.
- Zoom is disabled by default for stability.

**Geometry mapping:**
- Events use `mapToCube()` in `web/immersive/src/utils/geometry.ts`.
- The point cloud uses event `(W, R, u_τ)` to compute cube coordinates.

---

## 6) Left Column UI Modules

**Core components:**
- KPIs: `web/immersive/src/components/KPICards.tsx`
- Signal overview: `web/immersive/src/components/Sparklines.tsx`
- Geometry toggles: `web/immersive/src/components/GeometryControls.tsx`
- Generative controls: `web/immersive/src/components/GenerationControls.tsx`
- Playback controls: `web/immersive/src/components/Controls.tsx`

**KPIs**
- Time
- Events
- Cascade Probability
- Assets

**Signal Overview**
- λ(t), μ(t), ψ(t) only

**Geometry Controls**
- Show simplex overlay
- Point size toggle (Small / Medium / Large)

**Generative Controls**
- Trigger asset
- Radius (min R)
- Horizon (T hours)

**Debug / Verification**
- Build stamp + Mode/Playback indicators are displayed in Viewer to prove the correct bundle is running.

---

## 7) Event Rail (Right Column Timeline)

**File:** `web/immersive/src/components/EventRail.tsx`

This is a **minimal, non‑intrusive timeline** to avoid covering the cube.

**Encoding:**
- Dot position = event time
- Dot color = asset
- Dot size/glow = cascade probability ψ/λ

**Interactions:**
- Hover highlights the corresponding 3D event.
- Click jumps the playhead to the event time.

---

## 8) Generative Mode Behavior

**Goal:** continuation generation, analogous to a causal LLM.

**Flow:**
1. Seed from real history (conditioning context).
2. Select a trigger event (asset + min R).
3. Generate continuation until horizon T (hours).
4. Rebase time to start at 0.

**Play behavior:**
- If no generative run exists → Play triggers generation.
- If a generative run exists and is paused → Play resumes.
- If a generative run has ended → Play generates a new run.

**Backend endpoint:**
- `POST /generate` (alias of `/generate/continue`)

**Files:**
- `web/api/main.py`
- `cascades/simulate.py`
- `web/immersive/src/api/client.ts`

---

## 9) Data Flow and API

**Primary API endpoints:**
- `GET /runs`
- `GET /runs/{run_id}/meta`
- `GET /runs/{run_id}/events`
- `GET /runs/{run_id}/metrics`
- `POST /generate`

**Run artifacts:**
- `meta.json`
- `events.parquet`
- `metrics.parquet`

See `docs/immersive_viz.md` for the full data contract and export format.

---

## 10) Build Stamp + Debugging

The viewer includes a **build stamp** to confirm the frontend bundle is current. Console logs track:
- Mode changes
- Play/pause
- Generation start/end

This makes it easy to verify that changes are actually applied.

---

## 11) Customization Tips

**Layout**
- Modify layout in `web/immersive/src/pages/ViewerPage.tsx`.

**Colors / Theme**
- Tailwind config: `web/immersive/tailwind.config.js`.

**Typography**
- Global styles: `web/immersive/src/index.css`.

**Point size mapping**
- Point size values are set in `ViewerPage.tsx` and passed to `EventsPoints`.

**Simplex visibility**
- Simplex material and edges are defined in `SimplexPlane.tsx`.

---

## 12) Quickstart

**API**
```bash
cd web/api
uvicorn main:app --reload --port 8000
```

**Frontend**
```bash
cd web/immersive
npm run dev
```

**Override API URL**
```bash
VITE_API_URL=http://localhost:8000 npm run dev
```

---

## Related Docs

- `README.md` — repo overview and quickstart
- `readme_2.md` — mathematical model
- `docs/immersive_viz.md` — architecture and API schema

