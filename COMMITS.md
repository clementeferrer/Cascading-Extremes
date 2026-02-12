# Commit Log (Operational)

This file tracks high-impact commits so we can decide quickly whether to keep or revert.

## Safe Revert Workflow

Use non-destructive revert (preserves history):

```bash
git revert <COMMIT_SHA>
git push space main
```

Avoid `git reset --hard` on shared history.

---

## 5a12827d (Critical)

- Full SHA: `5a12827df16c0a72ae2a351520bf16467518337d`
- Message: `api: fix generative asset-axis mapping between model and UI order`
- Files: `web/api/main.py`

### Problem Before
- In `Generative`, a very negative BTC input could appear on ETH (or other swapped axis).
- Labels and generated values were misaligned.

### Root Cause
- Model internal asset order and UI/export asset order diverged:
  - Internal/model/CDF order was effectively `BNB, BTC, ETH`.
  - UI and run metadata were displayed as `BTC, ETH, BNB`.
- The pipeline used model output without explicit reordering on export.

### What This Commit Changed
- Added explicit symbol-order mapping utilities in API.
- Kept model inference in internal order.
- Reordered generated `W` columns to UI/config order before export.
- Stopped relying on a sorted asset list for generation inputs; now uses CDF file order explicitly.

### Impact
- Generated axes now match labels consistently.
- New generative runs after this commit should show BTC/ETH/BNB on correct tracks.

### Revert
```bash
git revert 5a12827d
git push space main
```

---

## 1a2c6c3b

- Full SHA: `1a2c6c3ba01a76887495997e241ed0510ad7f91b`
- Message: `viewer: add generative event-only returns panel`
- Files:
  - `web/immersive/src/components/ReturnsTracksPanel.tsx`
  - `web/immersive/src/pages/ViewerPage.tsx`

### Change
- Added returns panel to `Generative` mode using event-only projection (`R * W_i` per asset).
- Same visual style as Real panel.

### Revert
```bash
git revert 1a2c6c3b
git push space main
```

---

## 941925d8

- Full SHA: `941925d8b234f23790f50c7cecef4941dca5b877`
- Message: `viewer: set default generative horizon to 1000h`
- File: `web/immersive/src/pages/ViewerPage.tsx`

### Change
- Default `Horizon` in Generative controls changed from `240` to `1000`.

### Revert
```bash
git revert 941925d8
git push space main
```

---

## 9cb82961

- Full SHA: `9cb8296191aad950c3556f92f296cc7de36a2565`
- Message: `generative: condition on real context and add temperature controls`
- Files:
  - `second_phase/simulate.py`
  - `web/api/main.py`
  - `web/immersive/src/api/client.ts`
  - `web/immersive/src/components/GenerationControls.tsx`
  - `web/immersive/src/pages/ViewerPage.tsx`

### Change
- Generation switched to completion-from-real-context behavior.
- Added `temperature` control and sampling temperature plumbing.
- Improved numeric input UX in Generative controls.

### Revert
```bash
git revert 9cb82961
git push space main
```

---

## e94b48ef

- Full SHA: `e94b48ef2efc72d8367e745659af4b518a1cd4d5`
- Message: `Revert "phase2: switch to garch-standardized residuals for train and generation"`

### Why It Was Done
- Rolled back `8a17970b` because POP and Signal Overview became nearly flat.

### Revert (if needed)
```bash
git revert e94b48ef
git push space main
```

---

## c1ef5d16 (Critical)

- Full SHA: `c1ef5d16f85b2fcc77f6f751c0185a4ebc47a20f`
- Message: `returns: add generative SAITS imputation path with safe fallback`
- Files:
  - `web/api/returns.py`
  - `web/api/generative_imputation.py`
  - `web/api/tests/test_api.py`
  - `web/immersive/src/pages/ViewerPage.tsx`
  - `web/immersive/src/components/ReturnsTracksPanel.tsx`
  - `web/immersive/src/api/types.ts`
  - `second_phase/imputation_data.py`
  - `second_phase/train_imputer.py`
  - `requirements.txt`
  - `web/api/requirements.txt`
  - `artifacts/phase2/x_series.parquet`
  - `artifacts/phase2/x_to_returns_map.npz`

### Problem Before
- `GET /runs/{run_id}/returns` only worked for `real`.
- In `generative`, the time-series panel was frontend `event-only` projection and had no backend contract.

### What This Commit Changed
- Extended `/returns` to support `generative`.
- Added SAITS imputation pipeline in backend (hourly anchors from `X = R*W`).
- Added quantile calibration path from `X` to `% log-return`.
- Added safe fallback mode (`generative_event_only_fallback`) when SAITS/artifacts are unavailable.
- Unified frontend to always load panel data via `/returns` for both modes.
- Added optional response fields: `series_mode`, `imputation`.
- Added offline scripts/artifacts for imputation data preparation.

### Impact
- `Generative` panel now has a stable backend contract and degrades gracefully.
- If SAITS artifacts are missing, UX remains available via event-only fallback badge.

### Revert
```bash
git revert c1ef5d16
git push space main
```
