"""All Plotly visualization functions for the Cascade Explorer."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import plotly.graph_objects as go
import torch


# ── Unified palette ──────────────────────────────────────────────────────

_COLOR_REAL = "#264653"       # dark teal — real / primary data
_COLOR_GEN = "#e76f51"        # burnt orange — generated / secondary
_COLOR_ACCENT = "#2a9d8f"     # teal — reference lines, bounds, accents
_COLOR_NEUTRAL = "#475569"    # slate gray — neutral bars, grid

# EI colorscale: light (0) → red (1), fixed range [0, 1]
_EI_COLORSCALE = [[0, "#fee8c8"], [0.25, "#f4a582"], [0.5, "#e00000"], [1, "#b00000"]]


# ── Style helper ─────────────────────────────────────────────────────────


def _apply_style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Helvetica, Arial, sans-serif", size=14, color="#1a1a1a"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            font=dict(size=12, color="#1a1a1a"),
            bgcolor="white",
            bordercolor="#cccccc",
            borderwidth=1,
            x=1, xanchor="right",
            y=1, yanchor="bottom",
        ),
        margin=dict(l=60, r=20, t=50, b=50),
    )
    title_text = ""
    if fig.layout.title and fig.layout.title.text:
        title_text = fig.layout.title.text
    fig.update_layout(title=dict(
        text=title_text,
        font=dict(size=16, color="#1a1a1a"),
    ))
    fig.update_xaxes(
        title_font=dict(size=14, color="#1a1a1a"),
        tickfont=dict(size=12, color="#333333"),
        showgrid=True, gridcolor="#e5e5e5", gridwidth=1,
        zeroline=False,
    )
    fig.update_yaxes(
        title_font=dict(size=14, color="#1a1a1a"),
        tickfont=dict(size=12, color="#333333"),
        showgrid=True, gridcolor="#e5e5e5", gridwidth=1,
        zeroline=False,
    )
    return fig


# ── Downloadable chart helper ────────────────────────────────────────────

import streamlit as st  # noqa: E402


def downloadable_chart(fig: go.Figure, *, key: str):
    """Display a Plotly chart with a PNG download button underneath."""
    st.plotly_chart(fig, use_container_width=True)
    png_bytes = fig.to_image(format="png", width=1200, height=700, scale=3)
    title = (fig.layout.title.text if fig.layout.title and fig.layout.title.text else key)
    filename = title.lower().replace(" ", "_").replace(".", "") + ".png"
    st.download_button(
        f"Download {title}", data=png_bytes, file_name=filename,
        mime="image/png", key=key,
    )


# ── 3D Geometry primitives ───────────────────────────────────────────────


def _wireframe_sphere(n: int = 30) -> list:
    """Generate wireframe traces for a unit sphere."""
    traces = []
    # Latitude lines
    for phi in np.linspace(0, np.pi, 7):
        theta = np.linspace(0, 2 * np.pi, n)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.full_like(theta, np.cos(phi))
        traces.append(go.Scatter3d(
            x=x, y=y, z=z, mode="lines",
            line=dict(color="rgba(150,160,180,0.25)", width=1),
            showlegend=False, hoverinfo="skip",
        ))
    # Longitude lines
    for theta in np.linspace(0, 2 * np.pi, 13)[:-1]:
        phi = np.linspace(0, np.pi, n)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        traces.append(go.Scatter3d(
            x=x, y=y, z=z, mode="lines",
            line=dict(color="rgba(150,160,180,0.25)", width=1),
            showlegend=False, hoverinfo="skip",
        ))
    return traces


def _cube_frame_traces(half: float = 1.5) -> list:
    """Generate wireframe edges for a cube [-half, half]^3."""
    traces = []
    h = half
    # 12 edges of a cube
    edges = [
        ((-h,-h,-h), (h,-h,-h)), ((-h,-h,-h), (-h,h,-h)), ((-h,-h,-h), (-h,-h,h)),
        ((h,h,h), (-h,h,h)), ((h,h,h), (h,-h,h)), ((h,h,h), (h,h,-h)),
        ((-h,h,-h), (h,h,-h)), ((-h,h,-h), (-h,h,h)),
        ((h,-h,-h), (h,h,-h)), ((h,-h,-h), (h,-h,h)),
        ((-h,-h,h), (h,-h,h)), ((-h,-h,h), (-h,h,h)),
    ]
    for (a, b) in edges:
        traces.append(go.Scatter3d(
            x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]],
            mode="lines", line=dict(color="rgba(100,116,139,0.3)", width=1),
            showlegend=False, hoverinfo="skip",
        ))
    return traces


def _axis_traces(half: float = 1.5, labels: Optional[list] = None) -> list:
    """Axis lines from -half to +half with labels."""
    traces = []
    colors = ["#e76f51", "#2a9d8f", "#264653"]
    ax_labels = labels or ["X", "Y", "Z"]
    for i in range(3):
        start = [0, 0, 0]
        end = [0, 0, 0]
        end[i] = half
        neg = [0, 0, 0]
        neg[i] = -half
        traces.append(go.Scatter3d(
            x=[neg[i] if j == 0 else 0 for j in range(2)] if i == 0 else [0, 0],
            y=[neg[i] if j == 0 else 0 for j in range(2)] if i == 1 else [0, 0],
            z=[neg[i] if j == 0 else 0 for j in range(2)] if i == 2 else [0, 0],
            mode="lines", showlegend=False, hoverinfo="skip",
            line=dict(color=colors[i], width=2, dash="dot"),
        ))
        p = [0, 0, 0]
        p[i] = half
        n = [0, 0, 0]
        n[i] = -half
        # Positive axis
        traces.append(go.Scatter3d(
            x=[0, p[0]], y=[0, p[1]], z=[0, p[2]],
            mode="lines", showlegend=False, hoverinfo="skip",
            line=dict(color=colors[i], width=2),
        ))
        # Negative axis
        traces.append(go.Scatter3d(
            x=[0, n[0]], y=[0, n[1]], z=[0, n[2]],
            mode="lines", showlegend=False, hoverinfo="skip",
            line=dict(color=colors[i], width=2, dash="dot"),
        ))
        # Label
        lbl = [0, 0, 0]
        lbl[i] = half * 1.1
        traces.append(go.Scatter3d(
            x=[lbl[0]], y=[lbl[1]], z=[lbl[2]],
            mode="text", text=[ax_labels[i]],
            textfont=dict(size=12, color=colors[i]),
            showlegend=False, hoverinfo="skip",
        ))
    return traces


# ── 3D Scene (replicating web viewer) ────────────────────────────────────


def scene_3d_sphere(
    W: np.ndarray,
    R: np.ndarray,
    color_values: np.ndarray,
    labels: Optional[list] = None,
    show_sphere: bool = False,
    show_bulk: bool = False,
    bulk_positions: Optional[np.ndarray] = None,
    color_label: str = "EI",
    title: str = "3D cascade viewer",
    height: int = 600,
    colorscale: str = "Reds",
) -> go.Figure:
    """Full 3D scene: cube frame + axes + optional wireframe sphere + bulk cloud + extreme events at R*W."""
    fig = go.Figure()

    # Cube frame
    for t in _cube_frame_traces(1.5):
        fig.add_trace(t)

    # Coordinate axes
    for t in _axis_traces(1.5, labels):
        fig.add_trace(t)

    # Optional wireframe sphere
    if show_sphere:
        for t in _wireframe_sphere(25):
            fig.add_trace(t)

    # Bulk cloud (non-extreme observations)
    if show_bulk and bulk_positions is not None and len(bulk_positions) > 0:
        # Subsample for performance
        n_bulk = len(bulk_positions)
        if n_bulk > 5000:
            idx = np.random.choice(n_bulk, 5000, replace=False)
            bp = bulk_positions[idx]
        else:
            bp = bulk_positions
        fig.add_trace(go.Scatter3d(
            x=bp[:, 0], y=bp[:, 1], z=bp[:, 2],
            mode="markers",
            marker=dict(size=1.5, color="rgba(100,116,139,0.85)"),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Extreme events at R*W
    pos = R[:, None] * W
    hover = [
        f"Event {i}<br>R={R[i]:.3f}<br>W=({W[i,0]:.2f}, {W[i,1]:.2f}, {W[i,2]:.2f})"
        for i in range(len(R))
    ]
    is_ei = color_label == "EI"
    marker_opts = dict(
        size=5,
        color=color_values,
        colorscale=_EI_COLORSCALE if is_ei else colorscale,
        cmin=0.0 if is_ei else None,
        cmax=1.0 if is_ei else None,
        opacity=0.9,
        showscale=True,
        colorbar=dict(
            title=dict(text=color_label, font=dict(color="#1a1a1a")),
            tickfont=dict(color="#1a1a1a"),
        ),
    )
    fig.add_trace(go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode="markers",
        marker=marker_opts,
        text=hover,
        hoverinfo="text",
        name="Extreme events",
    ))

    fig.update_layout(
        height=height,
        scene=dict(
            xaxis_title=labels[0] if labels else "W1",
            yaxis_title=labels[1] if labels else "W2",
            zaxis_title=labels[2] if labels else "W3",
            bgcolor="white",
            aspectmode="cube",
            xaxis=dict(range=[-1.6, 1.6], gridcolor="#e5e5e5", showbackground=True, backgroundcolor="white"),
            yaxis=dict(range=[-1.6, 1.6], gridcolor="#e5e5e5", showbackground=True, backgroundcolor="white"),
            zaxis=dict(range=[-1.6, 1.6], gridcolor="#e5e5e5", showbackground=True, backgroundcolor="white"),
        ),
        title=title,
    )
    return _apply_style(fig)


# ── Intensity decomposition ──────────────────────────────────────────────


def intensity_decomposition_plot(
    T: np.ndarray,
    lam: np.ndarray,
    psi: np.ndarray,
) -> go.Figure:
    mu = lam - psi
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=T, y=lam, mode="lines", name="λ (total)",
        line=dict(color="#1f2937", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=T, y=mu, mode="lines", name="μ (exogenous)",
        line=dict(color="#94a3b8", width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=T, y=psi, mode="lines", name="ψ (endogenous)",
        line=dict(color="#e76f51", width=3),
        fill="tozeroy", fillcolor="rgba(231,111,81,0.15)",
    ))
    fig.update_layout(
        height=300,
        xaxis_title="Time (hours)",
        yaxis_title="Intensity",
        title="Intensity decomposition: λ = μ + ψ",
    )
    return _apply_style(fig)


# ── EI time series ───────────────────────────────────────────────────────


def poc_plot(
    T: np.ndarray,
    lam: np.ndarray,
    psi: np.ndarray,
) -> go.Figure:
    mu = np.maximum(lam - psi, 0.0)
    poc = 1.0 - mu / np.maximum(lam, 1e-8)
    mean_poc = float(np.mean(poc))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=T, y=poc, mode="lines", name="EI = 1 − μ/λ",
        line=dict(color="#e76f51", width=2.5),
    ))
    fig.add_hline(
        y=mean_poc, line_dash="dash", line_color="#475569",
        annotation_text=f"Mean EI = {mean_poc:.3f}",
    )
    fig.update_layout(
        height=280,
        xaxis_title="Time (hours)",
        yaxis_title="EI = 1 − μ(T) / λ(T)",
        yaxis_range=[0, 1],
        title="Endogeneity index (EI)",
    )
    return _apply_style(fig)


# ── Event rail ───────────────────────────────────────────────────────────


def event_rail_plot(
    T: np.ndarray,
    assets: np.ndarray,
    poc_values: Optional[np.ndarray] = None,
) -> go.Figure:
    """Event rail showing dominant asset per event over time."""
    fig = go.Figure()
    marker_opts = dict(size=8, color=_COLOR_REAL)
    if poc_values is not None:
        marker_opts = dict(
            size=8, color=poc_values, colorscale=_EI_COLORSCALE,
            cmin=0.0, cmax=1.0,
            showscale=True,
            colorbar=dict(title=dict(text="EI")),
        )
    fig.add_trace(go.Scatter(
        x=T, y=assets, mode="markers", name="Events", marker=marker_opts,
    ))
    fig.update_layout(
        height=240,
        xaxis_title="Time (hours)",
        yaxis_title="Dominant asset",
        title="Event timeline by dominant asset",
    )
    return _apply_style(fig)


# ── Returns tracks ───────────────────────────────────────────────────────


def returns_tracks_plot(
    W: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    symbols: list,
) -> go.Figure:
    """Plot R*W_j projections per asset over time."""
    fig = go.Figure()
    colors = ["#e76f51", "#2a9d8f", "#264653"]
    for j, sym in enumerate(symbols):
        if j < W.shape[1]:
            vals = R * W[:, j]
            fig.add_trace(go.Scatter(
                x=T, y=vals, mode="lines+markers",
                name=sym,
                line=dict(color=colors[j % len(colors)], width=2.5),
                marker=dict(size=4),
            ))
    fig.update_layout(
        height=300,
        xaxis_title="Time (hours)",
        yaxis_title="R · Wⱼ",
        title="Asset projections",
    )
    return _apply_style(fig)


# ── Subcriticality bar chart ─────────────────────────────────────────────


def subcriticality_bar_plot(
    row_sums: np.ndarray,
    margin: float = 0.95,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=row_sums, name="Row sum",
        marker_color=_COLOR_REAL,
    ))
    fig.add_hline(
        y=margin, line_dash="dash", line_color=_COLOR_GEN,
        annotation_text=f"Subcriticality bound ({margin})",
    )
    fig.update_layout(
        height=300,
        xaxis_title="Event index",
        yaxis_title="Kernel row sum",
        title="Subcriticality diagnostic",
    )
    return _apply_style(fig)


# ── Genealogy tree ───────────────────────────────────────────────────────


def genealogy_tree_plot(
    T: np.ndarray,
    parents: np.ndarray,
    cascade_probs: np.ndarray,
    title: str = "Genealogy tree",
) -> go.Figure:
    fig = go.Figure()
    n = len(parents)

    # Edges (parent -> child)
    for i in range(n):
        if parents[i] >= 0:
            j = int(parents[i])
            if j < len(T):
                fig.add_trace(go.Scatter(
                    x=[T[j], T[i]], y=[j, i],
                    mode="lines",
                    line=dict(color="rgba(100,116,139,0.4)", width=1.5),
                    showlegend=False, hoverinfo="skip",
                ))

    # Nodes
    fig.add_trace(go.Scatter(
        x=T[:n], y=np.arange(n),
        mode="markers",
        marker=dict(
            size=8, color=cascade_probs[:n], colorscale=_EI_COLORSCALE,
            cmin=0.0, cmax=1.0,
            showscale=True,
            colorbar=dict(title=dict(text="EI")),
        ),
        text=[f"Event {i}, Parent={parents[i]}, EI={cascade_probs[i]:.3f}" for i in range(n)],
        hoverinfo="text",
        name="Events",
    ))

    fig.update_layout(
        height=500,
        xaxis_title="Time (hours)",
        yaxis_title="Event index",
        title=title,
    )
    return _apply_style(fig)


# ── Cluster on sphere ────────────────────────────────────────────────────


def cluster_sphere_plot(
    W: np.ndarray,
    R: np.ndarray,
    cluster_indices: list,
    cluster_id: int,
    labels: Optional[list] = None,
) -> go.Figure:
    fig = go.Figure()
    # Wireframe sphere
    for t in _wireframe_sphere(15):
        fig.add_trace(t)

    idx = np.array(cluster_indices)
    # All events faint
    fig.add_trace(go.Scatter3d(
        x=W[:, 0], y=W[:, 1], z=W[:, 2],
        mode="markers", marker=dict(size=2, color="#cbd5e1", opacity=0.3),
        name="All events",
    ))
    # Cluster events connected
    fig.add_trace(go.Scatter3d(
        x=W[idx, 0], y=W[idx, 1], z=W[idx, 2],
        mode="markers+lines",
        marker=dict(size=5, color=R[idx], colorscale="Plasma", showscale=True,
                    colorbar=dict(title=dict(text="R"))),
        line=dict(color="rgba(231,111,81,0.6)", width=2),
        name=f"Cluster {cluster_id}",
    ))
    # Root
    fig.add_trace(go.Scatter3d(
        x=[W[idx[0], 0]], y=[W[idx[0], 1]], z=[W[idx[0], 2]],
        mode="markers",
        marker=dict(size=10, color="#e76f51", symbol="diamond"),
        name="Immigrant",
    ))

    xl = labels[0] if labels else "W1"
    yl = labels[1] if labels else "W2"
    zl = labels[2] if labels else "W3"
    fig.update_layout(
        height=480,
        scene=dict(
            xaxis_title=xl, yaxis_title=yl, zaxis_title=zl,
            bgcolor="white",
            aspectmode="cube",
            xaxis=dict(gridcolor="#e5e5e5", showbackground=True, backgroundcolor="white"),
            yaxis=dict(gridcolor="#e5e5e5", showbackground=True, backgroundcolor="white"),
            zaxis=dict(gridcolor="#e5e5e5", showbackground=True, backgroundcolor="white"),
        ),
        title=f"Cluster {cluster_id} ({len(idx)} events)",
    )
    return _apply_style(fig)


# ── Termination bounds ───────────────────────────────────────────────────


def termination_bounds_plot(
    nu: float,
    ns: np.ndarray,
    survival_bounds: np.ndarray,
    empirical_sizes: Optional[np.ndarray] = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ns, y=survival_bounds,
        mode="lines+markers", name=f"P(|C| ≥ n) ≤ ν<sup>n−1</sup>, ν = {nu:.3f}",
        line=dict(color="#e76f51", width=2),
    ))
    if empirical_sizes is not None and len(empirical_sizes) > 0:
        # Empirical survival function
        max_s = int(empirical_sizes.max())
        emp_ns = np.arange(1, max_s + 1)
        emp_surv = np.array([np.mean(empirical_sizes >= n) for n in emp_ns])
        fig.add_trace(go.Scatter(
            x=emp_ns, y=emp_surv,
            mode="lines+markers", name="Empirical P(|C| ≥ n)",
            line=dict(color="#2a9d8f", width=2, dash="dash"),
        ))
    fig.update_layout(
        height=350,
        xaxis_title="Cluster size n",
        yaxis_title="P(|C| ≥ n)",
        yaxis_type="log",
        title="Galton–Watson termination bounds",
    )
    return _apply_style(fig)


# ── Coexistence ──────────────────────────────────────────────────────────


def coexistence_plot(
    T: np.ndarray,
    coex: np.ndarray,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=T, y=coex, mode="lines",
        line=dict(color="#264653", width=2),
        fill="tozeroy", fillcolor="rgba(38,70,83,0.1)",
        name="Active cascades",
    ))
    fig.update_layout(
        height=280,
        xaxis_title="Time (hours)",
        yaxis_title="Active cascades",
        title=f"Cascade coexistence (max = {int(coex.max()) if len(coex) else 0})",
    )
    return _apply_style(fig)


# ── Attenuation heatmap ─────────────────────────────────────────────────


def attenuation_heatmap(
    W: np.ndarray,
    model,
    n_samples: int = 50,
) -> go.Figure:
    idx = np.random.choice(len(W), size=min(n_samples, len(W)), replace=False)
    idx = np.sort(idx)
    W_sub = W[idx]
    n = len(W_sub)

    W_i = torch.tensor(np.repeat(W_sub, n, axis=0), dtype=torch.float32)
    W_j = torch.tensor(np.tile(W_sub, (n, 1)), dtype=torch.float32)
    pair = torch.cat([W_i, W_j], dim=-1)

    with torch.no_grad():
        attn = torch.sigmoid(model.attenuation_net(pair)).numpy().reshape(n, n)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=attn, colorscale="Viridis",
        colorbar=dict(title=dict(text="κ")),
    ))
    fig.update_layout(
        height=400,
        xaxis_title="Direction j",
        yaxis_title="Direction i",
        title="Directional attenuation κ(Wᵢ, Wⱼ)",
    )
    return _apply_style(fig)


# ── Kernel heatmap ───────────────────────────────────────────────────────


def kernel_heatmap(kernel: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=kernel, colorscale="Viridis"))
    fig.update_layout(
        height=400,
        xaxis_title="Past event j",
        yaxis_title="Current event i",
        title="Hawkes kernel matrix Kᵢⱼ",
    )
    return _apply_style(fig)


# ── Radial comparison ────────────────────────────────────────────────────


def radial_comparison_plot(
    R_real: np.ndarray,
    R_gen: Optional[np.ndarray] = None,
) -> go.Figure:
    fig = go.Figure()
    # Compute shared bins so both histograms are directly comparable
    all_R = R_real if R_gen is None else np.concatenate([R_real, R_gen])
    bin_start = float(all_R.min())
    bin_end = float(all_R.max())
    bin_size = (bin_end - bin_start) / 40
    shared_bins = dict(start=bin_start, end=bin_end, size=bin_size)

    fig.add_trace(go.Histogram(
        x=R_real, name="Real", opacity=0.65, marker_color=_COLOR_REAL,
        histnorm="probability density",
        xbins=shared_bins,
    ))
    if R_gen is not None:
        fig.add_trace(go.Histogram(
            x=R_gen, name="Generated", opacity=0.65, marker_color=_COLOR_GEN,
            histnorm="probability density",
            xbins=shared_bins,
        ))
    fig.update_layout(
        barmode="overlay", height=300,
        title="Radial magnitude distribution",
        xaxis_title="R", yaxis_title="Density",
    )
    return _apply_style(fig)


# ── Gauge-rate scatter ───────────────────────────────────────────────────


def qq_plot(
    real_vals: np.ndarray,
    gen_vals: np.ndarray,
    label: str,
    n_quantiles: int = 100,
) -> go.Figure:
    """QQ-plot comparing quantiles of two distributions."""
    probs = np.linspace(0.01, 0.99, n_quantiles)
    q_real = np.quantile(real_vals, probs)
    q_gen = np.quantile(gen_vals, probs)
    lo = min(float(q_real.min()), float(q_gen.min()))
    hi = max(float(q_real.max()), float(q_gen.max()))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=q_real, y=q_gen, mode="markers",
        marker=dict(size=6, color=_COLOR_REAL),
        name="Quantiles",
    ))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color=_COLOR_ACCENT, dash="dash", width=2),
        name="y = x",
    ))
    fig.update_layout(
        height=350,
        xaxis_title=f"Real {label}",
        yaxis_title=f"Generated {label}",
        title=f"Q–Q plot: {label}",
    )
    return _apply_style(fig)


def acf_comparison_plot(
    acf_real: np.ndarray,
    acf_gen: np.ndarray,
    label: str,
) -> go.Figure:
    """Compare autocorrelation functions of real vs generated."""
    lags = np.arange(len(acf_real))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lags, y=acf_real, mode="lines+markers",
        line=dict(color=_COLOR_REAL, width=2.5),
        marker=dict(size=5), name="Real",
    ))
    fig.add_trace(go.Scatter(
        x=lags, y=acf_gen, mode="lines+markers",
        line=dict(color=_COLOR_GEN, width=2.5),
        marker=dict(size=5), name="Generated",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=_COLOR_NEUTRAL)
    fig.update_layout(
        height=300,
        xaxis_title="Lag",
        yaxis_title="Autocorrelation",
        title=f"{label} autocorrelation",
    )
    return _apply_style(fig)


def cluster_size_comparison_plot(
    sizes_real: np.ndarray,
    sizes_gen: np.ndarray,
) -> go.Figure:
    """Overlaid histogram of cluster sizes."""
    all_s = np.concatenate([sizes_real, sizes_gen])
    bin_start = float(all_s.min()) - 0.5
    bin_end = float(all_s.max()) + 0.5
    bin_size = max(1.0, (bin_end - bin_start) / 30)
    shared_bins = dict(start=bin_start, end=bin_end, size=bin_size)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=sizes_real, name="Real", opacity=0.6, marker_color=_COLOR_REAL,
        histnorm="probability density", xbins=shared_bins,
    ))
    fig.add_trace(go.Histogram(
        x=sizes_gen, name="Generated", opacity=0.6, marker_color=_COLOR_GEN,
        histnorm="probability density", xbins=shared_bins,
    ))
    fig.update_layout(
        barmode="overlay", height=300,
        title="Cluster size distribution",
        xaxis_title="Cluster size", yaxis_title="Density",
    )
    return _apply_style(fig)


def dominant_asset_bar_plot(
    freq_real: np.ndarray,
    freq_gen: np.ndarray,
    symbols: Sequence[str],
) -> go.Figure:
    """Grouped bar chart of dominant asset frequencies."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(symbols), y=freq_real, name="Real",
        marker_color=_COLOR_REAL,
    ))
    fig.add_trace(go.Bar(
        x=list(symbols), y=freq_gen, name="Generated",
        marker_color=_COLOR_GEN,
    ))
    fig.update_layout(
        barmode="group", height=300,
        title="Dominant asset frequency",
        xaxis_title="Asset", yaxis_title="Fraction",
        yaxis_range=[0, 1],
    )
    return _apply_style(fig)


def c2st_roc_plot(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_mean: float,
    auc_std: float,
) -> go.Figure:
    """ROC curve for the Classifier Two-Sample Test."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        line=dict(color=_COLOR_REAL, width=2.5),
        name=f"ROC (AUC = {auc_mean:.3f} +/- {auc_std:.3f})",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color=_COLOR_NEUTRAL, dash="dash", width=1),
        name="Random (AUC = 0.5)",
    ))
    fig.update_layout(
        height=350,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        title="Logistic classifier ROC — real vs generated",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
    )
    return _apply_style(fig)


def gauge_rate_scatter(
    R: np.ndarray,
    rates: np.ndarray,
    u: np.ndarray,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=R, y=rates, mode="markers",
        marker=dict(size=4, color=u, colorscale="Viridis", showscale=True,
                    colorbar=dict(title=dict(text="u<sub>τ</sub>"))),
        name="Events",
    ))
    fig.update_layout(
        height=300,
        xaxis_title="R",
        yaxis_title="Gauge rate β",
        title="Gauge rate β vs. magnitude R",
    )
    return _apply_style(fig)
