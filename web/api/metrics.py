from __future__ import annotations

from typing import Dict, List

import numpy as np


def summarize(events: List[Dict], metrics: List[Dict]) -> Dict:
    out: Dict = {}
    if metrics:
        lam = np.array([m.get("lambda") for m in metrics], dtype=float)
        psi = np.array([m.get("psi") for m in metrics], dtype=float)
        mean_mag = np.array([m.get("mean_mag") for m in metrics], dtype=float)
        event_rate = np.array([m.get("event_rate") for m in metrics], dtype=float)

        valid = np.isfinite(lam) & (lam > 0)
        if valid.any():
            out["branching_proxy"] = float(np.nanmean(psi[valid] / lam[valid]))
        if np.isfinite(mean_mag).any():
            out["mean_magnitude"] = float(np.nanmean(mean_mag))
        if np.isfinite(event_rate).any():
            out["event_rate"] = float(np.nanmean(event_rate))

    if events:
        mag = np.array([e.get("mag") for e in events], dtype=float)
        if np.isfinite(mag).any():
            out["mean_magnitude_raw"] = float(np.nanmean(mag))
            out["p95_magnitude"] = float(np.nanpercentile(mag, 95))
        out["event_count"] = len(events)
    return out
