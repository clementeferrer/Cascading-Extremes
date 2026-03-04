"""Page 3: Experimentation Lab — adjustable tau, generation experiments, statistics comparison."""

from __future__ import annotations

import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
import torch

from streamlit_test.data_loader import AppData
from streamlit_test.styles import hero_section, story_section
from streamlit_test.compute import (
    compute_hawkes_decomposition,
    compute_loss_components,
    build_genealogy_from_hawkes,
    build_real_context_prompt,
    run_autoregressive_generation,
    check_extremality,
    compute_autocorrelation,
    compute_dominant_freq,
    compute_transition_matrix,
    transition_matrix_distance,
    compute_c2st,
)
from streamlit_test.plots import (
    _apply_style,
    scene_3d_sphere,
    intensity_decomposition_plot,
    poc_plot,
    event_rail_plot,
    returns_tracks_plot,
    genealogy_tree_plot,
    radial_comparison_plot,
    qq_plot,
    acf_comparison_plot,
    dominant_asset_bar_plot,
    c2st_roc_plot,
)


def _subset(events: dict, n: int) -> dict:
    return {k: v[:n] for k, v in events.items()}


def _dt_comparison_plot(dT_real: np.ndarray, dT_gen: np.ndarray) -> go.Figure:
    """Overlay histogram of inter-event times (real vs generated)."""
    all_dT = np.concatenate([dT_real, dT_gen])
    bin_start = float(all_dT.min())
    bin_end = float(np.percentile(all_dT, 99))  # clip outliers for readability
    bin_size = (bin_end - bin_start) / 40
    shared_bins = dict(start=bin_start, end=bin_end, size=bin_size)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=dT_real, name="Real", opacity=0.65, marker_color="#264653",
        histnorm="probability density", xbins=shared_bins,
    ))
    fig.add_trace(go.Histogram(
        x=dT_gen, name="Generated", opacity=0.65, marker_color="#e76f51",
        histnorm="probability density", xbins=shared_bins,
    ))
    fig.update_layout(
        barmode="overlay", height=300,
        title="Inter-event time distribution",
        xaxis_title="ΔT (hours)", yaxis_title="Density",
    )
    return _apply_style(fig)


def render_experiment(data: AppData):
    hero_section(
        "Experimentation Lab",
        "Adjust parameters and compare real vs generated statistics",
        "Threshold | Comparison",
    )

    tabs = st.tabs(["Threshold Effect", "Statistics Comparison"])

    symbols = data.symbols

    # ── Tab 1: Threshold Effect ──────────────────────────────────────
    with tabs[0]:
        _render_threshold_effect(data, symbols)

    # ── Tab 2: Statistics Comparison ─────────────────────────────────
    with tabs[1]:
        _render_statistics_comparison(data, symbols)


def _render_threshold_effect(data: AppData, symbols):
    story_section(
        "Threshold Effect",
        "Adjust tau to see how the number of extreme events changes. Higher tau = fewer, more extreme events.",
    )

    tau_slider = st.slider(
        "Quantile level tau", 0.90, 0.99, data.tau, 0.01,
        key="tau_slider",
    )

    events = data.real_events
    W, R = events["W"], events["R"]

    # Compute thresholds at current model tau
    W_t = torch.tensor(W, dtype=torch.float32)
    with torch.no_grad():
        u_current = data.q_model(W_t).numpy()

    # Approximate threshold at new tau by scaling
    # u_new ≈ u_current * (log(1/(1-tau_new)) / log(1/(1-tau_current)))
    # This is a rough approximation since the true threshold depends on quantile regression
    scale = np.log(1.0 / (1.0 - tau_slider + 1e-10)) / np.log(1.0 / (1.0 - data.tau + 1e-10))
    u_approx = u_current * scale

    # Filter events above approximate threshold
    mask = R > u_approx
    n_extreme = int(mask.sum())
    n_total = len(R)

    c1, c2, c3 = st.columns(3)
    c1.metric("tau", f"{tau_slider:.2f}")
    c2.metric("Extreme events", f"{n_extreme} / {n_total}")
    c3.metric("Extreme %", f"{100 * n_extreme / n_total:.1f}%")

    if n_extreme > 0 and data.d_assets == 3:
        W_ext = W[mask]
        R_ext = R[mask]
        # Compute EI for coloring (same as Viewer)
        lam_th, psi_th, _ = compute_hawkes_decomposition(
            data, events["T"], R, W, events["dT"],
            window=min(1024, len(events["T"])), tokens=events.get("tokens"),
        )
        if lam_th is not None:
            mu_th = np.maximum(lam_th - psi_th, 0.0)
            ei_full = np.zeros(len(R))
            start_th = max(0, len(R) - len(lam_th))
            ei_full[start_th:] = 1.0 - mu_th / np.maximum(lam_th, 1e-8)
            ei_ext = ei_full[mask]
            color_vals, color_lbl = ei_ext, "EI"
        else:
            color_vals, color_lbl = R_ext, "R"
        st.plotly_chart(
            scene_3d_sphere(
                W_ext, R_ext, color_vals,
                labels=symbols, color_label=color_lbl,
                title=f"Extreme events at τ = {tau_slider:.2f} (n = {n_extreme})",
            ),
            use_container_width=True,
        )

    # Show threshold distribution
    fig_u = go.Figure()
    fig_u.add_trace(go.Histogram(x=u_current, name=f"u<sub>τ</sub> (τ = {data.tau})", opacity=0.65, marker_color="#264653"))
    fig_u.add_trace(go.Histogram(x=u_approx, name=f"u<sub>τ</sub> (τ = {tau_slider})", opacity=0.65, marker_color="#e76f51"))
    fig_u.update_layout(barmode="overlay", height=280, title="Threshold distribution", xaxis_title="u<sub>τ</sub>(W)")
    st.plotly_chart(_apply_style(fig_u), use_container_width=True)


def _spherical_to_w(theta: float, phi: float) -> np.ndarray:
    """Convert spherical coordinates to unit direction on S^2."""
    return np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ], dtype=np.float32)


def _w_to_spherical(w: np.ndarray) -> tuple:
    """Cartesian W -> (theta, phi) spherical coordinates."""
    phi = float(np.arccos(np.clip(w[2], -1, 1)))
    theta = float(np.arctan2(w[1], w[0]))
    if theta < 0:
        theta += 2 * np.pi
    return theta, phi


def _spherical_direction_input(symbols, key_prefix: str, data: AppData | None = None):
    """Render spherical coordinate inputs with asset presets. Returns (w0, label)."""
    # Presets: canonical asset directions (+/-)
    # Positive Laplace margin = rally, Negative = crash
    presets = {}

    # First real extreme preset (when data is available)
    if data is not None and len(data.real_events.get("W", [])) > 0:
        W_first = data.real_events["W"][0]
        R_first = float(data.real_events["R"][0])
        theta_first, phi_first = _w_to_spherical(W_first)
        presets[f"First real extreme (R={R_first:.2f})"] = (theta_first, phi_first)

    presets.update({
        f"{symbols[0]} (+)": (0.0, np.pi / 2),
        f"{symbols[0]} (-)": (np.pi, np.pi / 2),
        f"{symbols[1]} (+)": (np.pi / 2, np.pi / 2),
        f"{symbols[1]} (-)": (3 * np.pi / 2, np.pi / 2),
        f"{symbols[2]} (+)": (0.0, 0.0),
        f"{symbols[2]} (-)": (0.0, np.pi),
        "Custom": None,
    })
    preset = st.selectbox("Direction preset", list(presets.keys()), key=f"{key_prefix}_preset")

    # When preset changes, push its angles (and magnitude for first-event) into session state
    prev_key = f"{key_prefix}_prev_preset"
    if preset != st.session_state.get(prev_key):
        st.session_state[prev_key] = preset
        if presets[preset] is not None:
            st.session_state[f"{key_prefix}_theta"] = presets[preset][0]
            st.session_state[f"{key_prefix}_phi"] = presets[preset][1]
        # Auto-fill magnitude when selecting the first real extreme
        if "First real extreme" in preset and data is not None:
            R_first = float(data.real_events["R"][0])
            st.session_state[f"{key_prefix}_shock_mag"] = R_first

    # Set defaults only if not already in session state
    if f"{key_prefix}_theta" not in st.session_state:
        st.session_state[f"{key_prefix}_theta"] = 0.0
    if f"{key_prefix}_phi" not in st.session_state:
        st.session_state[f"{key_prefix}_phi"] = np.pi / 2

    c1, c2 = st.columns(2)
    with c1:
        theta = st.number_input(
            "θ (azimuthal, 0–2π)",
            min_value=0.0, max_value=2 * np.pi, step=0.1,
            key=f"{key_prefix}_theta", format="%.2f",
        )
    with c2:
        phi = st.number_input(
            "φ (polar, 0–π)",
            min_value=0.0, max_value=np.pi, step=0.1,
            key=f"{key_prefix}_phi", format="%.2f",
        )

    w0 = _spherical_to_w(theta, phi)
    st.caption(f"W = ({w0[0]:.3f}, {w0[1]:.3f}, {w0[2]:.3f})")
    return w0, preset


def _render_statistics_comparison(data: AppData, symbols):
    story_section(
        "Statistics Comparison",
        "Compare real vs simulated event statistics side by side.",
    )

    real = data.real_events

    # ── Source selector ──────────────────────────────────────────────
    source = st.radio(
        "Simulation source",
        ["Fresh generation (recommended)", "Stored simulation"],
        horizontal=True,
        key="stat_cmp_source",
    )

    sim = None

    if source == "Fresh generation (recommended)":
        st.markdown(
            "> **Fresh generation** uses `autoregressive_generate()` with the "
            "**full real event history as prompt context** — the same approach "
            "used by the 3D Viewer's generative mode. This produces statistically "
            "comparable events because the Transformer is conditioned on the "
            "entire real history."
        )

        w0, preset_label = _spherical_direction_input(symbols, "stat", data)

        c1, c2, c3 = st.columns(3)
        with c1:
            shock_mag = st.number_input(
                "Magnitude R", value=3.0, min_value=0.1, step=0.5, key="stat_shock_mag",
            )
        with c2:
            horizon = st.number_input(
                "Horizon (hours)", value=240.0, step=50.0,
                min_value=10.0, max_value=5000.0, key="stat_horizon",
            )
        with c3:
            temperature = st.slider(
                "Temperature", min_value=0.2, max_value=3.0, value=1.0, step=0.1,
                key="stat_temperature",
                help="Controls sampling diversity. <1 = sharper (more concentrated), >1 = more diffuse.",
            )

        if st.button("Generate Fresh Comparison", key="stat_gen_btn"):
            r0 = float(shock_mag)
            is_extreme, u_tau = check_extremality(w0, r0, data.q_model)
            if not is_extreme:
                r0 = u_tau + 0.1  # clamp above threshold

            # Build prompt from full real history
            prompt = build_real_context_prompt(real)

            with st.spinner(
                f"Generating cascade from {preset_label} "
                f"(R={r0:.2f}, horizon={horizon:.0f}h) "
                f"with {len(real['T'])} real events as context..."
            ):
                gen_events = run_autoregressive_generation(
                    w0, r0, horizon, data, temperature=temperature, prompt=prompt,
                )

            st.session_state["stat_cmp_fresh"] = gen_events
            st.session_state["stat_cmp_fresh_info"] = {
                "asset": preset_label,
                "mag": shock_mag,
                "r0": r0,
                "horizon": horizon,
                "n_context": len(real["T"]),
                "temperature": temperature,
            }

        # Use cached result if available
        if "stat_cmp_fresh" in st.session_state:
            sim = st.session_state["stat_cmp_fresh"]
            info = st.session_state.get("stat_cmp_fresh_info", {})
            n_gen = len(sim["T"])
            # Skip seed event (first element is the prompt seed)
            if n_gen > 1:
                st.success(
                    f"Generated **{n_gen - 1}** continuation events "
                    f"(+ 1 seed) from {info.get('asset', '?')} shock "
                    f"(R={info.get('r0', 0):.2f}), "
                    f"conditioned on {info.get('n_context', '?')} real events."
                )
            elif n_gen == 1:
                st.warning("Only the seed event was generated — try a longer horizon or different shock.")
        else:
            st.info("Click **Generate Fresh Comparison** to produce simulated events with full context.")

    else:  # Stored simulation
        st.warning(
            "The stored simulation (`simulated_events.npz`) was generated with only "
            "20 seed events and Ogata thinning — it represents a **single short cascade**, "
            "not an equilibrium distribution comparable to the 2-year real data. "
            "Metrics will appear mismatched. Use **Fresh generation** for a fair comparison."
        )
        sim = data.sim_events
        if sim is None:
            st.info("No stored simulation found. Run `python -m second_phase.simulate` first.")
            return

    if sim is None:
        return

    # ── Comparison metrics ───────────────────────────────────────────
    _render_comparison_body(data, real, sim, symbols)


def _render_comparison_body(data: AppData, real, sim, symbols):
    """Shared comparison body for both fresh and stored simulations."""

    # Basic metrics table
    metrics = {
        "Metric": ["Events", "Mean R", "Std R", "Min R", "Max R", "Mean dT (h)"],
        "Real": [
            len(real["R"]),
            f"{real['R'].mean():.3f}",
            f"{real['R'].std():.3f}",
            f"{real['R'].min():.3f}",
            f"{real['R'].max():.3f}",
            f"{real['dT'].mean():.2f}",
        ],
        "Simulated": [
            len(sim["R"]),
            f"{sim['R'].mean():.3f}",
            f"{sim['R'].std():.3f}",
            f"{sim['R'].min():.3f}",
            f"{sim['R'].max():.3f}",
            f"{sim['dT'].mean():.2f}",
        ],
    }
    st.dataframe(metrics, use_container_width=True)

    # ── QQ-Plots ─────────────────────────────────────────────────────
    fig_qq_r = qq_plot(real["R"], sim["R"], "R")
    fig_qq_dt = qq_plot(real["dT"], sim["dT"], "dT")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_qq_r, use_container_width=True)
    with c2:
        st.plotly_chart(fig_qq_dt, use_container_width=True)

    # ── C2ST: Classifier Two-Sample Test ─────────────────────────────
    story_section(
        "C2ST: Classifier Two-Sample Test",
        "Can a logistic regression classifier distinguish real from generated events? "
        "AUC ~ 0.5 = indistinguishable (good), AUC > 0.7 = distinguishable (bad).",
    )
    # Build feature vectors: [R, log(dT), W...] per event
    d = len(symbols)
    real_feat = np.column_stack([real["R"], np.log(real["dT"] + 1e-8), real["W"]])
    gen_feat = np.column_stack([sim["R"], np.log(sim["dT"] + 1e-8), sim["W"]])
    if len(real_feat) >= 10 and len(gen_feat) >= 10:
        c2st = compute_c2st(real_feat, gen_feat)
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_roc = c2st_roc_plot(c2st["fpr"], c2st["tpr"], c2st["auc_mean"], c2st["auc_std"])

            st.plotly_chart(fig_roc, use_container_width=True)
        with c2:
            st.metric("C2ST AUC", f"{c2st['auc_mean']:.3f} +/- {c2st['auc_std']:.3f}")
            if c2st["auc_mean"] < 0.6:
                st.success("Excellent: distributions are nearly indistinguishable.")
            elif c2st["auc_mean"] < 0.7:
                st.info("Good: mild separability.")
            else:
                st.warning("Separable: the classifier can distinguish real from generated.")
    else:
        st.info("Not enough events for C2ST (need >= 10 per class).")

    # Compute Hawkes decomposition for both
    real_window = min(1024, len(real["T"]))
    sim_window = min(1024, len(sim["T"]))
    real_lam, real_psi, real_kernel = compute_hawkes_decomposition(
        data, real["T"], real["R"], real["W"], real["dT"],
        window=real_window, tokens=real.get("tokens"),
    )
    sim_lam, sim_psi, sim_kernel = compute_hawkes_decomposition(
        data, sim["T"], sim["R"], sim["W"], sim["dT"],
        window=sim_window, tokens=sim.get("tokens"),
    )

    # EI metrics
    real_poc_arr, sim_poc_arr = None, None
    if real_lam is not None and sim_lam is not None:
        real_mu = np.maximum(real_lam - real_psi, 0.0)
        real_poc_arr = 1.0 - real_mu / np.maximum(real_lam, 1e-8)
        sim_mu = np.maximum(sim_lam - sim_psi, 0.0)
        sim_poc_arr = 1.0 - sim_mu / np.maximum(sim_lam, 1e-8)

        c1, c2 = st.columns(2)
        c1.metric("Real Mean EI", f"{float(np.mean(real_poc_arr)):.3f}")
        c2.metric("Generated Mean EI", f"{float(np.mean(sim_poc_arr)):.3f}")

        # Branching ratio comparison
        if real_kernel is not None and sim_kernel is not None:
            nu_real = float(real_kernel.sum(axis=-1).mean())
            nu_gen = float(sim_kernel.sum(axis=-1).mean())
            c1, c2 = st.columns(2)
            c1.metric("Real branching ratio nu", f"{nu_real:.4f}")
            c2.metric("Generated branching ratio nu", f"{nu_gen:.4f}")

    # ── Distribution comparisons ─────────────────────────────────────
    # Magnitude boxplots
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(y=real["R"], name="Real", marker_color="#264653"))
    fig_box.add_trace(go.Box(y=sim["R"], name="Generated", marker_color="#e76f51"))
    fig_box.update_layout(height=300, title="Magnitude distribution")
    _apply_style(fig_box)

    st.plotly_chart(fig_box, use_container_width=True)

    # Magnitude histograms
    fig_radial = radial_comparison_plot(real["R"], sim["R"])
    fig_dt = _dt_comparison_plot(real["dT"], sim["dT"])

    st.plotly_chart(fig_radial, use_container_width=True)
    st.plotly_chart(fig_dt, use_container_width=True)

    # ── dT Autocorrelation comparison ────────────────────────────────
    acf_real = compute_autocorrelation(real["dT"])
    acf_gen = compute_autocorrelation(sim["dT"])
    fig_acf = acf_comparison_plot(acf_real, acf_gen, "dT")

    st.plotly_chart(fig_acf, use_container_width=True)

    # EI distribution overlay
    if real_poc_arr is not None and sim_poc_arr is not None:
        fig_ei = go.Figure()
        fig_ei.add_trace(go.Histogram(
            x=real_poc_arr, name="Real", opacity=0.65, marker_color="#264653",
            histnorm="probability density", nbinsx=30,
        ))
        fig_ei.add_trace(go.Histogram(
            x=sim_poc_arr, name="Generated", opacity=0.65, marker_color="#e76f51",
            histnorm="probability density", nbinsx=30,
        ))
        fig_ei.update_layout(
            barmode="overlay", height=300,
            title="Endogeneity index distribution", xaxis_title="EI = 1 − μ/λ", yaxis_title="Density",
        )
        _apply_style(fig_ei)

        st.plotly_chart(fig_ei, use_container_width=True)

    # ── Intensity decomposition (side-by-side) ───────────────────────
    if real_lam is not None and sim_lam is not None:
        real_start = max(0, len(real["T"]) - len(real_lam))
        sim_start = max(0, len(sim["T"]) - len(sim_lam))
        fig_int_real = intensity_decomposition_plot(real["T"][real_start:], real_lam, real_psi)
        fig_int_sim = intensity_decomposition_plot(sim["T"][sim_start:], sim_lam, sim_psi)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Real**")
            st.plotly_chart(fig_int_real, use_container_width=True)
        with c2:
            st.markdown("**Generated**")
            st.plotly_chart(fig_int_sim, use_container_width=True)

    # ── EI time series (side-by-side) ────────────────────────────────
    if real_lam is not None and sim_lam is not None:
        fig_poc_real = poc_plot(real["T"][real_start:], real_lam, real_psi)
        fig_poc_sim = poc_plot(sim["T"][sim_start:], sim_lam, sim_psi)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_poc_real, use_container_width=True)
        with c2:
            st.plotly_chart(fig_poc_sim, use_container_width=True)

    # ── Genealogy Tree (side-by-side) ────────────────────────────────
    if real_lam is not None and sim_lam is not None:
        real_gen = build_genealogy_from_hawkes(real_lam, real_psi, real_kernel)
        sim_gen = build_genealogy_from_hawkes(sim_lam, sim_psi, sim_kernel)
        real_start = max(0, len(real["T"]) - len(real_lam))
        sim_start = max(0, len(sim["T"]) - len(sim_lam))
        fig_gen_real = genealogy_tree_plot(real["T"][real_start:], real_gen.parents, real_gen.cascade_probs)
        fig_gen_sim = genealogy_tree_plot(sim["T"][sim_start:], sim_gen.parents, sim_gen.cascade_probs)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Real**")
            st.plotly_chart(fig_gen_real, use_container_width=True)
        with c2:
            st.markdown("**Generated**")
            st.plotly_chart(fig_gen_sim, use_container_width=True)

    # ── Event Rail (side-by-side) ────────────────────────────────────
    real_dominant = np.argmax(np.abs(real["W"]), axis=1)
    real_asset_labels = np.array([symbols[i] if i < len(symbols) else str(i) for i in real_dominant])
    sim_dominant = np.argmax(np.abs(sim["W"]), axis=1)
    sim_asset_labels = np.array([symbols[i] if i < len(symbols) else str(i) for i in sim_dominant])

    real_rail_poc, sim_rail_poc = None, None
    if real_poc_arr is not None:
        real_rail_poc = np.zeros(len(real["T"]))
        real_rail_poc[max(0, len(real["T"]) - len(real_poc_arr)):] = real_poc_arr
    if sim_poc_arr is not None:
        sim_rail_poc = np.zeros(len(sim["T"]))
        sim_rail_poc[max(0, len(sim["T"]) - len(sim_poc_arr)):] = sim_poc_arr

    fig_rail_real = event_rail_plot(real["T"], real_asset_labels, real_rail_poc)
    fig_rail_sim = event_rail_plot(sim["T"], sim_asset_labels, sim_rail_poc)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_rail_real, use_container_width=True)
    with c2:
        st.plotly_chart(fig_rail_sim, use_container_width=True)

    # ── Returns tracks (side-by-side) ────────────────────────────────
    fig_ret_real = returns_tracks_plot(real["W"], real["R"], real["T"], symbols)
    fig_ret_sim = returns_tracks_plot(sim["W"], sim["R"], sim["T"], symbols)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_ret_real, use_container_width=True)
    with c2:
        st.plotly_chart(fig_ret_sim, use_container_width=True)

    # ── Side-by-side 3D ──────────────────────────────────────────────
    if data.d_assets == 3:
        c1, c2 = st.columns(2)
        # Build full-length EI arrays for coloring
        if real_poc_arr is not None:
            real_ei_full = np.zeros(len(real["T"]))
            real_ei_full[max(0, len(real["T"]) - len(real_poc_arr)):] = real_poc_arr
            real_color, real_clbl = real_ei_full, "EI"
        else:
            real_color, real_clbl = real["R"], "R"
        if sim_poc_arr is not None:
            sim_ei_full = np.zeros(len(sim["T"]))
            sim_ei_full[max(0, len(sim["T"]) - len(sim_poc_arr)):] = sim_poc_arr
            sim_color, sim_clbl = sim_ei_full, "EI"
        else:
            sim_color, sim_clbl = sim["R"], "R"
        fig_3d_real = scene_3d_sphere(
            real["W"], real["R"], real_color,
            labels=symbols, color_label=real_clbl, title="Real", height=450,
        )
        fig_3d_sim = scene_3d_sphere(
            sim["W"], sim["R"], sim_color,
            labels=symbols, color_label=sim_clbl, title="Generated", height=450,
        )

        with c1:
            st.plotly_chart(fig_3d_real, use_container_width=True)
        with c2:
            st.plotly_chart(fig_3d_sim, use_container_width=True)

    # ── Loss Diagnostics ─────────────────────────────────────────────
    story_section(
        "Loss Diagnostics",
        "Per-component log-likelihoods (direction, magnitude, time) on real vs generated events.",
    )
    real_loss = compute_loss_components(
        data, real["T"], real["R"], real["W"], real["dT"],
        u=real.get("u"), tokens=real.get("tokens"),
    )
    sim_loss = compute_loss_components(
        data, sim["T"], sim["R"], sim["W"], sim["dT"],
        u=sim.get("u"), tokens=sim.get("tokens"),
    )
    if real_loss is not None and sim_loss is not None:
        comp_names = ["Direction (W)", "Magnitude (R)", "Time (dT)"]
        comp_keys = ["log_p_w", "log_p_r", "log_p_t"]
        real_vals = [real_loss[k] for k in comp_keys]
        sim_vals = [sim_loss[k] for k in comp_keys]

        fig_diag = go.Figure()
        fig_diag.add_trace(go.Bar(
            name="Real", x=comp_names, y=real_vals,
            marker_color="#264653",
        ))
        fig_diag.add_trace(go.Bar(
            name="Generated", x=comp_names, y=sim_vals,
            marker_color="#e76f51",
        ))
        fig_diag.update_layout(
            barmode="group", height=350,
            title="Per-component mean log-likelihood",
            yaxis_title="Mean log p",
        )
        _apply_style(fig_diag)

        st.plotly_chart(fig_diag, use_container_width=True)

        # Numeric summary
        diag_table = {
            "Component": comp_names,
            "Real": [f"{v:.3f}" for v in real_vals],
            "Generated": [f"{v:.3f}" for v in sim_vals],
            "Gap": [f"{r - s:.3f}" for r, s in zip(real_vals, sim_vals)],
        }
        st.dataframe(diag_table, use_container_width=True)
    else:
        st.info("Not enough events to compute loss diagnostics.")

    # ── Transition Matrix (side-by-side) ─────────────────────────────
    d = len(symbols)
    P_real = compute_transition_matrix(real["W"], d)
    P_gen = compute_transition_matrix(sim["W"], d)
    frob = transition_matrix_distance(P_real, P_gen)
    st.metric("Transition Matrix Frobenius Distance", f"{frob:.4f}")

    def _transition_heatmap(prob, labels, title):
        text = [[f"{prob[i, j]:.2f}" for j in range(len(labels))] for i in range(len(labels))]
        fig = ff.create_annotated_heatmap(
            z=prob, x=labels, y=labels, annotation_text=text,
            colorscale=[[0, "#fee8c8"], [0.5, "#f4a582"], [1, "#e76f51"]],
            showscale=True,
        )
        fig.update_layout(
            title=title, height=350,
            xaxis_title="To", yaxis_title="From",
            xaxis_side="bottom",
        )
        for ann in fig.layout.annotations:
            ann.font = dict(size=13)
        return _apply_style(fig)

    fig_tm_real = _transition_heatmap(P_real, symbols, "Real transition matrix")
    fig_tm_sim = _transition_heatmap(P_gen, symbols, "Generated transition matrix")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_tm_real, use_container_width=True)
    with c2:
        st.plotly_chart(fig_tm_sim, use_container_width=True)

    # ── Dominant Asset Frequency ─────────────────────────────────────
    freq_real = compute_dominant_freq(real["W"], d)
    freq_gen = compute_dominant_freq(sim["W"], d)
    fig_dom = dominant_asset_bar_plot(freq_real, freq_gen, symbols)

    st.plotly_chart(fig_dom, use_container_width=True)

