"""Page 2: Theory & Genealogy — directed forest, Galton-Watson bounds, subcriticality."""

from __future__ import annotations

import numpy as np
import streamlit as st

from streamlit_test.data_loader import AppData
from streamlit_test.styles import hero_section, story_section
from streamlit_test.compute import (
    compute_hawkes_decomposition,
    build_genealogy_from_hawkes,
    compute_parent_probs,
    compute_subcriticality_diagnostics,
    compute_coexistence,
    cascade_termination_bounds,
)
from streamlit_test.plots import (
    genealogy_tree_plot,
    cluster_sphere_plot,
    subcriticality_bar_plot,
    termination_bounds_plot,
    coexistence_plot,
    kernel_heatmap,
    attenuation_heatmap,
)


def _subset(events: dict, n: int) -> dict:
    return {k: v[:n] for k, v in events.items()}


def render_theory(data: AppData):
    hero_section(
        "Theory & Genealogy",
        "Immigration-branching structure, directed forests, Galton-Watson bounds",
        "Hawkes genealogy | Subcriticality | Coexistence",
    )

    # ── Sidebar controls ─────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("Theory Controls")

    T, R, W, dT = data.real_events["T"], data.real_events["R"], data.real_events["W"], data.real_events["dT"]

    n_events = len(T)
    default_window = min(n_events, 1024)
    if "theory_window" not in st.session_state or st.session_state["theory_window"] < default_window:
        st.session_state["theory_window"] = default_window
    window = st.sidebar.slider("Transformer context window", 20, 1024, key="theory_window")
    tokens = data.real_events.get("tokens")
    symbols = data.symbols

    # Compute Hawkes decomposition
    lam, psi, kernel = compute_hawkes_decomposition(
        data, T, R, W, dT, window=window, tokens=tokens,
    )

    if lam is None or kernel is None:
        st.warning("Not enough events for genealogy analysis.")
        return

    start = max(0, len(T) - len(lam))
    T_w = T[start:]
    R_w = R[start:]
    W_w = W[start:]

    # Build genealogy
    genealogy = build_genealogy_from_hawkes(lam, psi, kernel)
    n_total = len(genealogy.parents)
    n_imm = int(genealogy.immigrant_mask.sum())
    n_clusters = len(genealogy.clusters)
    cluster_sizes = [len(v) for v in genealogy.clusters.values()]
    mean_csize = float(np.mean(cluster_sizes)) if cluster_sizes else 0.0
    max_csize = int(np.max(cluster_sizes)) if cluster_sizes else 0
    endogeneity = float(np.mean(genealogy.cascade_probs))

    # KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Events", str(n_total))
    c2.metric("Immigrants", f"{n_imm} ({100*n_imm/max(n_total,1):.0f}%)")
    c3.metric("Clusters", str(n_clusters))
    c4.metric("Mean |C|", f"{mean_csize:.1f}")
    c5.metric("Max |C|", str(max_csize))
    c6.metric("Endogeneity", f"{endogeneity:.3f}")

    # ── Tabs ─────────────────────────────────────────────────────────
    tabs = st.tabs([
        "Genealogy Tree", "Cluster Explorer", "Subcriticality",
        "Termination Bounds", "Coexistence", "Parent Probs", "Kernel Analysis",
    ])

    # Tab 1: Genealogy Tree
    with tabs[0]:
        story_section("Directed Forest", "Each event is either an immigrant (exogenous) or child of a parent. Edges show parent -> child links.")
        st.plotly_chart(
            genealogy_tree_plot(T_w, genealogy.parents, genealogy.cascade_probs),
            use_container_width=True,
        )
        # EI distribution
        import plotly.graph_objects as go
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=genealogy.cascade_probs, nbinsx=30, marker_color="#264653"))
        fig_hist.update_layout(height=280, xaxis_title="1 - mu/lambda", title="EI Distribution")
        from streamlit_test.plots import _apply_style
        st.plotly_chart(_apply_style(fig_hist), use_container_width=True)

    # Tab 2: Cluster Explorer
    with tabs[1]:
        story_section("Cluster Explorer", "Select a cluster to inspect its BFS chain on the sphere.")
        sorted_by_size = sorted(
            genealogy.clusters.items(), key=lambda x: -len(x[1])
        )
        cluster_options = [f"Cluster {k} ({len(v)} events)" for k, v in sorted_by_size]

        if cluster_options:
            selected = st.selectbox("Select cluster", cluster_options, key="cluster_sel")
            cluster_id = int(selected.split(" ")[1])
            cluster_indices = genealogy.clusters[cluster_id]

            if data.d_assets == 3:
                st.plotly_chart(
                    cluster_sphere_plot(W_w, R_w, cluster_indices, cluster_id, labels=symbols),
                    use_container_width=True,
                )

            # Active interval
            t_start = T_w[cluster_indices[0]]
            t_end = T_w[cluster_indices[-1]]
            st.write(f"**Active interval:** {t_start:.1f}h — {t_end:.1f}h (duration: {t_end - t_start:.1f}h)")

            # Event table
            st.markdown(f"**Events in cluster {cluster_id}:**")
            rows = []
            for idx in cluster_indices:
                parent = genealogy.parents[idx]
                cp = genealogy.cascade_probs[idx]
                w_str = ", ".join(f"{w:.3f}" for w in W_w[idx])
                rows.append({
                    "Index": idx,
                    "T (h)": f"{T_w[idx]:.1f}",
                    "R": f"{R_w[idx]:.3f}",
                    "W": f"({w_str})",
                    "Parent": int(parent),
                    "EI": f"{cp:.3f}",
                })
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No clusters found.")

    # Tab 3: Subcriticality
    with tabs[2]:
        story_section("Subcriticality Diagnostic", "Kernel row sums must stay below the margin to ensure finite cluster sizes.")
        diag = compute_subcriticality_diagnostics(kernel)
        margin = data.cfg["model"].get("subcrit_margin", 0.95)

        c1, c2, c3 = st.columns(3)
        c1.metric("Max row sum", f"{diag['max_row_sum']:.4f}")
        c2.metric("Spectral radius", f"{diag['spectral_radius']:.4f}")
        c3.metric("Margin", f"{margin}")

        penalty = max(0, diag["max_row_sum"] - margin) ** 2
        st.write(f"**Penalty value:** {penalty:.6f}")

        st.plotly_chart(
            subcriticality_bar_plot(diag["row_sums"], margin),
            use_container_width=True,
        )

    # Tab 4: Termination Bounds
    with tabs[3]:
        story_section(
            "Galton-Watson Termination Bounds",
            "For subcritical branching: E[|C|] <= 1/(1-nu), P(|C|>=n) <= nu^{n-1}.",
        )
        # Estimate nu from mean offspring
        mu_gw = np.maximum(lam - psi, 0.0)
        mean_offspring = float(np.mean(1.0 - mu_gw / np.maximum(lam, 1e-8)))
        nu = min(mean_offspring, 0.999)
        bounds = cascade_termination_bounds(nu, max_n=50)

        c1, c2, c3 = st.columns(3)
        c1.metric("nu (mean offspring)", f"{nu:.3f}")
        c2.metric("E[|C|] bound", f"{bounds['expected_size']:.2f}")
        c3.metric("Empirical mean |C|", f"{mean_csize:.2f}")

        # Empirical cluster sizes for comparison
        empirical_sizes = np.array(cluster_sizes, dtype=np.float64)
        st.plotly_chart(
            termination_bounds_plot(nu, bounds["ns"], bounds["survival_prob"], empirical_sizes),
            use_container_width=True,
        )

        # Cluster size histogram
        import plotly.graph_objects as go
        fig_sz = go.Figure()
        fig_sz.add_trace(go.Histogram(x=empirical_sizes, nbinsx=20, marker_color="#264653"))
        fig_sz.update_layout(height=280, xaxis_title="Cluster size", title="Empirical Cluster Size Distribution")
        from streamlit_test.plots import _apply_style
        st.plotly_chart(_apply_style(fig_sz), use_container_width=True)

    # Tab 5: Coexistence
    with tabs[4]:
        story_section("Cascade Coexistence", "Number of simultaneously active cascades over time.")
        T_coex, coex = compute_coexistence(T_w, genealogy.parents, genealogy.clusters)
        c1, c2 = st.columns(2)
        c1.metric("Max coexistence", str(int(coex.max())) if len(coex) else "0")
        c2.metric("Mean coexistence", f"{float(coex.mean()):.2f}" if len(coex) else "0")
        st.plotly_chart(coexistence_plot(T_coex, coex), use_container_width=True)

    # Tab 6: Parent Probabilities
    with tabs[5]:
        story_section("Parent Probabilities", "P(parent=j | event i) heatmap + immigration probability.")
        parent_probs = compute_parent_probs(lam, psi, kernel)

        # Immigration probability bar chart
        import plotly.graph_objects as go
        from streamlit_test.plots import _apply_style
        imm_prob = parent_probs[:, 0]
        fig_imm = go.Figure()
        fig_imm.add_trace(go.Bar(y=imm_prob, marker_color="#264653"))
        fig_imm.add_hline(y=0.5, line_dash="dash", line_color="#e76f51")
        fig_imm.update_layout(
            height=280, xaxis_title="Event index",
            yaxis_title="P(immigrant)", title="Immigration Probability",
        )
        st.plotly_chart(_apply_style(fig_imm), use_container_width=True)

        # Full parent prob heatmap (limit to last 100 for readability)
        n_show = min(100, parent_probs.shape[0])
        pp_sub = parent_probs[-n_show:, :n_show + 1]
        fig_pp = go.Figure()
        fig_pp.add_trace(go.Heatmap(z=pp_sub, colorscale="Viridis"))
        fig_pp.update_layout(
            height=400, xaxis_title="Parent (0=immigrant, 1..n=event)",
            yaxis_title="Event i", title="Parent Probability Matrix P(parent=j | i)",
        )
        st.plotly_chart(_apply_style(fig_pp), use_container_width=True)

    # Tab 7: Kernel Analysis
    with tabs[6]:
        story_section("Kernel Analysis", "Hawkes excitation matrix K[i,j] and directional attenuation.")
        st.plotly_chart(kernel_heatmap(kernel), use_container_width=True)
        if data.d_assets == 3:
            st.plotly_chart(
                attenuation_heatmap(W_w, data.model, n_samples=50),
                use_container_width=True,
            )
