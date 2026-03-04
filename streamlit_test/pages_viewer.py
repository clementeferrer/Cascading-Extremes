"""Page 1: 3D Immersive Viewer — replicates the web viewer in Streamlit."""

from __future__ import annotations

import numpy as np
import streamlit as st

from streamlit_test.data_loader import AppData
from streamlit_test.styles import hero_section, story_section, panel
from streamlit_test.compute import compute_hawkes_decomposition
from streamlit_test.plots import (
    scene_3d_sphere,
    intensity_decomposition_plot,
    poc_plot,
    event_rail_plot,
    returns_tracks_plot,
)


def _subset(events: dict, n: int) -> dict:
    return {k: v[:n] for k, v in events.items()}


def render_viewer(data: AppData):
    hero_section(
        "3D Immersive Viewer",
        "Generative Cascades of Multivariate Extremes - Laboratory",
        "",
    )

    # ── Sidebar controls ─────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("Viewer Controls")
    show_bulk = st.sidebar.checkbox("Show bulk cloud", value=True)
    n_events = len(data.real_events["T"])
    default_window = min(n_events, 1024)
    influence_window = st.sidebar.slider("Transformer context window", 20, 1024, default_window)

    symbols = data.symbols
    events = data.real_events
    n_total = len(events["T"])

    # Playback slider
    n_show = st.slider("Playback — events to show", 10, n_total, n_total, step=10)
    ev = _subset(events, n_show)

    T, R, W, dT = ev["T"], ev["R"], ev["W"], ev["dT"]
    tokens = ev.get("tokens")

    # KPI row
    lam, psi, kernel = compute_hawkes_decomposition(
        data, T, R, W, dT, window=influence_window, tokens=tokens,
    )
    mean_poc = 0.0
    poc_arr = None
    if lam is not None and len(lam) > 0:
        mu_arr = np.maximum(lam - psi, 0.0)
        poc_arr = 1.0 - mu_arr / np.maximum(lam, 1e-8)
        mean_poc = float(np.mean(poc_arr))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Events", str(n_show))
    c2.metric("Mean EI", f"{mean_poc:.3f}")
    c3.metric("Mean R", f"{R.mean():.3f}")
    c4.metric("tau", f"{data.tau}")
    c5.metric("Assets", ", ".join(symbols))

    # 3D scene
    if lam is not None:
        start = max(0, len(T) - len(lam))
        full_poc = np.zeros(len(T))
        full_poc[start:] = poc_arr
        color_values = full_poc
        color_label = "EI"
    else:
        color_values = R
        color_label = "R"

    fig_3d = scene_3d_sphere(
        W, R, color_values, labels=symbols,
        show_bulk=show_bulk,
        bulk_positions=data.bulk_positions,
        color_label=color_label,
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    # Tabs
    tab_int, tab_poc, tab_rail, tab_ret = st.tabs(
        ["Intensity", "EI", "Event Rail", "Returns"]
    )

    with tab_int:
        if lam is not None:
            start = max(0, len(T) - len(lam))
            st.plotly_chart(
                intensity_decomposition_plot(T[start:], lam, psi),
                use_container_width=True,
            )
        else:
            st.info("Not enough events for intensity decomposition.")

    with tab_poc:
        if lam is not None:
            start = max(0, len(T) - len(lam))
            st.plotly_chart(poc_plot(T[start:], lam, psi), use_container_width=True)

    with tab_rail:
        dominant = np.argmax(np.abs(W), axis=1)
        asset_labels = np.array([symbols[i] if i < len(symbols) else str(i) for i in dominant])
        poc_vals = None
        if lam is not None:
            start = max(0, len(T) - len(lam))
            poc_full = np.zeros(len(T))
            poc_full[start:] = poc_arr
            poc_vals = poc_full
        st.plotly_chart(event_rail_plot(T, asset_labels, poc_vals), use_container_width=True)

    with tab_ret:
        st.plotly_chart(returns_tracks_plot(W, R, T, symbols), use_container_width=True)
