"""Phase 2 Streamlit Dashboard: Full-Sphere Cascading Extremes with Laplace Margins.

Three modes: Story, Analyst, Genealogy.
Run: streamlit run app_phase2.py
"""

from pathlib import Path
from datetime import datetime
import json
import textwrap

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from second_phase.dataset import spherical_features, build_token_sphere, compute_radial_angular_l2
from second_phase.model import SphericalCascadeTransformer, ModelConfig
from second_phase.extremes import SphericalQuantileMLP, QuantileModelConfig
from second_phase.genealogy import build_genealogy, Genealogy
from second_phase.simulate import (
    load_model,
    load_quantile_model,
    ogata_thinning,
    generate_with_genealogy,
    prompt_generate,
)
from second_phase.utils import load_config, ensure_dir


# ── Styling (reused from app.py) ──────────────────────────────────────────

def inject_styles():
    st.markdown(
        """
        <style>
        :root { color-scheme: light; }
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Newsreader:wght@400;600&display=swap');
        header[data-testid="stHeader"] {
            display: none !important;
        }
        .stApp {
            background: linear-gradient(120deg, #f7f4ee 0%, #eef3ff 100%);
        }
        html, body, [class*="css"]  {
            font-family: 'Space Grotesk', sans-serif;
            color: #0f172a;
        }
        .stMarkdown, .stText, p, span, label, div {
            color: #0f172a;
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e2e8f0;
        }
        [data-testid="stSidebar"] * {
            color: #0f172a;
        }
        .stButton button,
        .stButton button *,
        .stButton button p,
        .stButton button span,
        .stButton button div {
            background: #0f172a !important;
            color: #ffffff !important;
            border-color: #0f172a !important;
        }
        .stButton button:hover,
        .stButton button:hover * {
            background: #111827 !important;
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] div[data-baseweb="select"] > div {
            background-color: #f8fafc;
            color: #0f172a;
            border-color: #e2e8f0;
        }
        [data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea {
            background-color: #f8fafc;
            color: #0f172a;
            border-color: #e2e8f0;
        }
        [data-testid="stSidebar"] .stSlider {
            color: #0f172a;
        }
        [data-testid="stSidebar"] [data-baseweb="slider"] * {
            color: #0f172a;
        }
        div[data-baseweb="select"] > div {
            background-color: #f8fafc;
            color: #0f172a;
            border-color: #e2e8f0;
        }
        div[data-baseweb="popover"] {
            background-color: #ffffff !important;
            color: #0f172a !important;
        }
        div[data-baseweb="popover"] * {
            color: #0f172a !important;
        }
        div[data-baseweb="menu"] {
            background-color: #ffffff !important;
            color: #0f172a !important;
        }
        div[data-baseweb="menu"] * {
            color: #0f172a !important;
        }
        div[data-baseweb="menu"] li {
            background-color: #ffffff !important;
        }
        div[data-baseweb="menu"] li:hover {
            background-color: #f1f5f9 !important;
        }
        div[data-baseweb="menu"] ul {
            background-color: #ffffff !important;
        }
        div[role="listbox"] {
            background-color: #ffffff;
            color: #0f172a;
        }
        ul[role="listbox"] {
            background-color: #ffffff !important;
            color: #0f172a !important;
        }
        div[role="option"] {
            color: #0f172a;
        }
        li[role="option"] {
            background-color: #ffffff !important;
            color: #0f172a !important;
        }
        div[role="option"][aria-selected="true"] {
            background-color: #e2e8f0;
        }
        div[role="option"]:hover {
            background-color: #f1f5f9;
        }
        div[role="option"] span {
            color: #0f172a !important;
        }
        input, textarea, select {
            background-color: #f8fafc !important;
            color: #0f172a !important;
            border-color: #e2e8f0 !important;
        }
        h1, h2, h3, h4 {
            font-family: 'Newsreader', serif;
            letter-spacing: 0.2px;
        }
        .hero {
            background: linear-gradient(135deg, #1e3a5f 0%, #2d1b4e 100%);
            color: #f8fafc;
            padding: 28px 32px;
            border-radius: 16px;
            margin-bottom: 18px;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.2);
        }
        .hero-title {
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .hero-sub {
            font-size: 16px;
            opacity: 0.9;
        }
        .hero-tag {
            margin-top: 10px;
            display: inline-block;
            font-size: 12px;
            letter-spacing: 0.6px;
            text-transform: uppercase;
            background: rgba(255,255,255,0.12);
            padding: 6px 10px;
            border-radius: 999px;
        }
        .hero, .hero * {
            color: #f8fafc;
        }
        .panel {
            background: #ffffff;
            border-radius: 14px;
            padding: 16px 18px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
        }
        .panel h4 {
            margin: 0 0 6px 0;
            font-size: 16px;
        }
        .panel p {
            margin: 0;
            font-size: 13px;
            line-height: 1.5;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border-radius: 12px;
            padding: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 6px 14px rgba(15, 23, 42, 0.08);
        }
        .section-block {
            background: #ffffff;
            border-radius: 16px;
            padding: 18px 20px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
            margin: 10px 0 12px 0;
        }
        .section-headline {
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .section-sub {
            font-size: 13px;
            color: #475569;
        }
        .rule {
            height: 1px;
            background: linear-gradient(90deg, rgba(15,23,42,0.2), rgba(15,23,42,0.0));
            margin: 12px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero_section(title, subtitle, tag):
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">{title}</div>
            <div class="hero-sub">{subtitle}</div>
            <div class="hero-tag">{tag}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def story_section(title, subtitle):
    html = f"""
    <div class="section-block">
        <div class="section-headline">{title}</div>
        <div class="section-sub">{subtitle}</div>
    </div>
    """
    st.markdown(textwrap.dedent(html), unsafe_allow_html=True)


def apply_fig_style(fig):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Space Grotesk", size=12, color="#0f172a"),
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(255,255,255,0.0)",
        legend=dict(
            font=dict(color="#0f172a"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#e2e8f0",
            borderwidth=1,
        ),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    title_text = ""
    if fig.layout.title and fig.layout.title.text:
        title_text = fig.layout.title.text
    fig.update_layout(title=dict(text=title_text, font=dict(color="#0f172a")))
    fig.update_xaxes(title_font=dict(color="#0f172a"), tickfont=dict(color="#0f172a"))
    fig.update_yaxes(title_font=dict(color="#0f172a"), tickfont=dict(color="#0f172a"))
    return fig


# ── Loading helpers ───────────────────────────────────────────────────────

@st.cache_resource
def load_phase2_model(artifact_dir: str):
    return load_model(str(Path(artifact_dir) / "model.pt"))


@st.cache_resource
def load_phase2_qmodel(artifact_dir: str, d_assets: int, hidden_sizes):
    cfg = QuantileModelConfig(hidden_sizes=tuple(hidden_sizes))
    return load_quantile_model(str(Path(artifact_dir) / "quantile_model.pt"), d_assets, cfg)


def ensure_events(events):
    if "tokens" not in events or events["tokens"] is None:
        W, R, dT = events["W"], events["R"], events["dT"]
        xi = spherical_features(W).astype(np.float32)
        tokens = np.concatenate(
            [W, xi, np.log(R + 1e-8)[:, None], np.log(dT + 1e-8)[:, None]],
            axis=1,
        ).astype(np.float32)
        events["tokens"] = tokens
    if "u" not in events:
        events["u"] = np.ones_like(events["R"])
    return events


def subset_events(events, n):
    return {k: v[:n] for k, v in events.items()}


# ── Visualization helpers ─────────────────────────────────────────────────

def _wireframe_sphere(n=30):
    """Generate wireframe mesh for a unit sphere."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def sphere_scatter(W, color_values, color_label="R", title="Events on S^{d-1}", labels=None):
    """3D scatter of events on the unit sphere, colored by a scalar."""
    fig = go.Figure()

    # Wireframe sphere
    x_s, y_s, z_s = _wireframe_sphere(20)
    fig.add_trace(go.Surface(
        x=x_s, y=y_s, z=z_s,
        opacity=0.08,
        colorscale=[[0, "rgb(200,200,220)"], [1, "rgb(200,200,220)"]],
        showscale=False,
        hoverinfo="skip",
    ))

    xl = labels[0] if labels and len(labels) >= 3 else "W1"
    yl = labels[1] if labels and len(labels) >= 3 else "W2"
    zl = labels[2] if labels and len(labels) >= 3 else "W3"

    fig.add_trace(go.Scatter3d(
        x=W[:, 0], y=W[:, 1], z=W[:, 2],
        mode="markers",
        marker=dict(
            size=4,
            color=color_values,
            colorscale="Viridis",
            opacity=0.85,
            showscale=True,
            colorbar=dict(title=dict(text=color_label, font=dict(color="#0f172a")), tickfont=dict(color="#0f172a")),
        ),
        name="Events",
    ))

    fig.update_layout(
        height=500,
        scene=dict(
            xaxis_title=xl, yaxis_title=yl, zaxis_title=zl,
            bgcolor="rgba(255,255,255,0.0)",
            aspectmode="cube",
        ),
        title=title,
    )
    return apply_fig_style(fig)


def timeline_plot(T, color_values=None, color_label="psi/lambda", spotlight_idx=None, title="Event Timeline"):
    fig = go.Figure()
    marker_opts = dict(size=5, color="#475569")
    if color_values is not None:
        marker_opts = dict(
            size=5, color=color_values, colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=dict(text=color_label)),
        )
    fig.add_trace(go.Scatter(
        x=T, y=np.zeros_like(T), mode="markers", name="Events", marker=marker_opts,
    ))
    if spotlight_idx is not None and len(spotlight_idx) > 0:
        fig.add_trace(go.Scatter(
            x=T[spotlight_idx], y=np.zeros(len(spotlight_idx)),
            mode="markers", name="Spotlight",
            marker=dict(size=9, color="#e76f51"),
        ))
    fig.update_layout(height=260, yaxis=dict(showticklabels=False, title=""), xaxis_title="Time (hours)", title=title)
    return apply_fig_style(fig)


def intensity_plot(T, lam, psi, attenuation=None):
    mu = lam - psi
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=T, y=lam, mode="lines", name="lambda", line=dict(color="#1f2937", width=3)))
    fig.add_trace(go.Scatter(x=T, y=mu, mode="lines", name="mu (exogenous)", line=dict(color="#94a3b8", width=2, dash="dash")))
    fig.add_trace(go.Scatter(
        x=T, y=psi, mode="lines", name="psi (endogenous)",
        line=dict(color="#e76f51", width=3),
        fill="tozeroy", fillcolor="rgba(231,111,81,0.15)",
    ))
    fig.update_layout(height=300, xaxis_title="Time (hours)", title="Intensity Decomposition")
    return apply_fig_style(fig)


def genealogy_tree_plot(T, parents, cascade_probs, title="Genealogy Tree"):
    """Directed graph of parent-child relationships."""
    fig = go.Figure()
    n = len(parents)

    # Edges
    for i in range(n):
        if parents[i] >= 0:
            j = int(parents[i])
            fig.add_trace(go.Scatter(
                x=[T[j], T[i]], y=[j, i],
                mode="lines",
                line=dict(color="rgba(100,116,139,0.4)", width=1),
                showlegend=False, hoverinfo="skip",
            ))

    # Nodes
    colors = cascade_probs
    fig.add_trace(go.Scatter(
        x=T, y=np.arange(n),
        mode="markers",
        marker=dict(
            size=6, color=colors, colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=dict(text="psi/lambda")),
        ),
        text=[f"Event {i}, P={parents[i]}, cp={cascade_probs[i]:.2f}" for i in range(n)],
        hoverinfo="text",
        name="Events",
    ))

    fig.update_layout(
        height=max(400, n * 3),
        xaxis_title="Time (hours)",
        yaxis_title="Event index",
        title=title,
    )
    return apply_fig_style(fig)


def cluster_plot(T, W, R, cluster_indices, cluster_id, labels=None):
    """Show a single cluster on the sphere and timeline."""
    fig = go.Figure()
    x_s, y_s, z_s = _wireframe_sphere(15)
    fig.add_trace(go.Surface(
        x=x_s, y=y_s, z=z_s, opacity=0.06,
        colorscale=[[0, "rgb(200,200,220)"], [1, "rgb(200,200,220)"]],
        showscale=False, hoverinfo="skip",
    ))

    xl = labels[0] if labels and len(labels) >= 3 else "W1"
    yl = labels[1] if labels and len(labels) >= 3 else "W2"
    zl = labels[2] if labels and len(labels) >= 3 else "W3"

    idx = np.array(cluster_indices)
    # All events faint
    fig.add_trace(go.Scatter3d(
        x=W[:, 0], y=W[:, 1], z=W[:, 2],
        mode="markers", marker=dict(size=2, color="#cbd5e1", opacity=0.3),
        name="All events",
    ))
    # Cluster events
    fig.add_trace(go.Scatter3d(
        x=W[idx, 0], y=W[idx, 1], z=W[idx, 2],
        mode="markers+lines",
        marker=dict(size=5, color=R[idx], colorscale="Plasma", showscale=True,
                    colorbar=dict(title=dict(text="R"))),
        line=dict(color="rgba(231,111,81,0.6)", width=2),
        name=f"Cluster {cluster_id}",
    ))
    # Root highlighted
    fig.add_trace(go.Scatter3d(
        x=[W[idx[0], 0]], y=[W[idx[0], 1]], z=[W[idx[0], 2]],
        mode="markers", marker=dict(size=10, color="#e76f51", symbol="diamond"),
        name="Immigrant",
    ))

    fig.update_layout(
        height=480,
        scene=dict(xaxis_title=xl, yaxis_title=yl, zaxis_title=zl, bgcolor="rgba(255,255,255,0.0)", aspectmode="cube"),
        title=f"Cluster {cluster_id} ({len(idx)} events)",
    )
    return apply_fig_style(fig)


def subcriticality_plot(kernel):
    """Bar chart of kernel row sums vs bound."""
    row_sums = kernel.sum(axis=-1)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=row_sums, name="Row sum", marker_color="#475569"))
    fig.add_hline(y=0.95, line_dash="dash", line_color="#e76f51", annotation_text="Subcrit bound (0.95)")
    fig.update_layout(height=300, xaxis_title="Event index", yaxis_title="Kernel row sum", title="Subcriticality Diagnostic")
    return apply_fig_style(fig)


def attenuation_heatmap(W, model, n_samples=50, labels=None):
    """Visualize learned attenuation kappa(W_i, W_j) for sample directions."""
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
    fig.add_trace(go.Heatmap(z=attn, colorscale="Viridis", colorbar=dict(title=dict(text="kappa"))))
    fig.update_layout(height=400, xaxis_title="Direction j", yaxis_title="Direction i", title="Attenuation Heatmap kappa(W_i, W_j)")
    return apply_fig_style(fig)


def radial_histogram(R_real, R_gen=None, title="Radial Magnitude Distribution"):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=R_real, name="Real", opacity=0.6, marker_color="#475569"))
    if R_gen is not None:
        fig.add_trace(go.Histogram(x=R_gen, name="Generated", opacity=0.6, marker_color="#e76f51"))
    fig.update_layout(barmode="overlay", height=300, title=title)
    return apply_fig_style(fig)


def laplace_margin_histogram(X, symbols):
    """Histogram of Laplace-margined values to verify symmetric heavy tails."""
    fig = go.Figure()
    for i, col in enumerate(symbols):
        if i < X.shape[1]:
            fig.add_trace(go.Histogram(x=X[:, i], name=col, opacity=0.5))
    fig.update_layout(barmode="overlay", height=300, title="Laplace Margin Histogram", xaxis_title="X (Laplace)")
    return apply_fig_style(fig)


# ── Hawkes computation (model inference) ──────────────────────────────────

def compute_hawkes(model, events, window=128):
    events = ensure_events(events)
    tokens = events["tokens"]
    R = events["R"]
    T = events["T"]
    W = events["W"]

    if len(tokens) < 3:
        return None, None, None

    max_len = getattr(model.cfg, "max_len", window)
    window = min(window, max_len, len(tokens))
    start = max(0, len(tokens) - window)
    tokens = tokens[start:]
    R = R[start:]
    T = T[start:]
    W = W[start:]

    tokens_t = torch.tensor(tokens[None, :, :], dtype=torch.float32)
    T_tensor = torch.tensor(T[None, :], dtype=torch.float32)
    R_tensor = torch.tensor(R[None, :], dtype=torch.float32)
    W_tensor = torch.tensor(W[None, :, :], dtype=torch.float32)
    log_r = torch.log(R_tensor + 1e-8)

    with torch.no_grad():
        lam, psi, kernel = model.hawkes_intensity(
            model.encode(tokens_t), T_tensor, log_r, W=W_tensor, return_kernel=True,
        )

    return (
        lam.squeeze(0).numpy(),
        psi.squeeze(0).numpy(),
        kernel.squeeze(0).numpy(),
    )


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Phase 2: Full-Sphere Cascading Extremes", layout="wide")
    inject_styles()
    hero_section(
        "Phase 2: Full-Sphere Cascading Extremes",
        "Laplace margins, von Mises-Fisher directions, Ogata thinning, immigration-branching genealogy",
        "EVT x Sphere x Genealogy",
    )

    cfg_path = "configs/phase2.yaml"
    if not Path(cfg_path).exists():
        st.warning("Config not found at configs/phase2.yaml.")
        return

    cfg = load_config(cfg_path)
    artifact_dir = cfg.get("artifact_dir", "artifacts/phase2")

    if not Path(artifact_dir, "model.pt").exists():
        st.warning(f"Model not found at {artifact_dir}/model.pt. Train first with: python -m second_phase.train --config configs/phase2.yaml")
        return

    model = load_phase2_model(artifact_dir)
    symbols = cfg["data"]["symbols"]

    # Load meta
    meta_path = Path(artifact_dir) / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
            symbols = meta.get("symbols", symbols)

    d_assets = model.d_assets
    q_model = load_phase2_qmodel(
        artifact_dir, d_assets,
        cfg["extremes"]["quantile_model"]["hidden_sizes"],
    )

    # Load events
    events_path = Path("data/processed_phase2/events.npz")
    if not events_path.exists():
        st.warning("Processed events not found. Run training first.")
        return
    real = ensure_events(dict(np.load(events_path)))

    sim_path = Path(artifact_dir) / "simulated_events.npz"
    sim = None
    if sim_path.exists():
        sim = ensure_events(dict(np.load(sim_path)))

    gen_path = Path(artifact_dir) / "genealogy.npz"
    gen_data = None
    if gen_path.exists():
        gen_data = dict(np.load(gen_path))

    # ── Sidebar ───────────────────────────────────────────────────────
    st.sidebar.header("Director Controls")
    mode = st.sidebar.selectbox("Mode", ["Story", "Analyst", "Genealogy"])
    n_events = st.sidebar.slider("Max events", 50, 2000, 500, 50)
    influence_window = st.sidebar.slider("Influence window", 20, 200, 120)
    dataset_choice = st.sidebar.selectbox("Dataset", ["Real", "Generated"])

    chosen = real if dataset_choice == "Real" or sim is None else sim

    # ── Story mode ────────────────────────────────────────────────────
    if mode == "Story":
        story_section(
            "Story Mode: How Cascades Form on the Sphere",
            "Full R^d analysis: Laplace margins capture crashes AND rallies with sign information.",
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Assets", ", ".join(symbols))
        col2.metric("Tau", f"{cfg['extremes']['tau']:.2f}")
        col3.metric("Total events", f"{len(chosen['T'])}")
        col4.metric("Norm", "L2 (sphere)")

        st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)
        col_a.markdown(
            '<div class="panel"><h4>Data</h4><p>Hourly returns with GARCH filtering and PIT to standard Laplace margins. Values can be negative (rallies) or positive (crashes).</p></div>',
            unsafe_allow_html=True,
        )
        col_b.markdown(
            '<div class="panel"><h4>Model</h4><p>Causal transformer with von Mises-Fisher direction, truncated Gamma magnitude, Hawkes + directional attenuation time.</p></div>',
            unsafe_allow_html=True,
        )
        col_c.markdown(
            '<div class="panel"><h4>Genealogy</h4><p>Immigration-branching structure: each event is either an immigrant (exogenous) or triggered by a parent. Subcriticality ensures finite clusters.</p></div>',
            unsafe_allow_html=True,
        )

        ev = subset_events(chosen, n_events)

        # Laplace margin check (if we have the raw Laplace data)
        cdfs_path = Path(artifact_dir) / "cdfs.npz"
        if cdfs_path.exists():
            story_section("Laplace Margins", "Symmetric, heavy-tailed, centered at 0.")
            # Reconstruct X from events W and R for visualization
            X_approx = ev["W"] * ev["R"][:, None]
            st.plotly_chart(laplace_margin_histogram(X_approx, symbols), use_container_width=True)

        # Sphere scatter
        story_section("Geometry: Events on the Sphere", "Direction W lives on S^{d-1}. Negative components capture rallies.")
        if d_assets == 3:
            st.plotly_chart(
                sphere_scatter(ev["W"], ev["R"], color_label="R", title="Events on the Unit Sphere", labels=symbols),
                use_container_width=True,
            )

        # Timeline
        story_section("Timeline: When Extremes Occur", "Dense bursts indicate self-excitation cascades.")
        st.plotly_chart(timeline_plot(ev["T"], title=f"{dataset_choice} Event Timeline"), use_container_width=True)

        # Intensity decomposition
        story_section("Causal Pressure: Endogenous vs Exogenous", "When psi dominates lambda, the cascade is self-exciting.")
        lam, psi, kernel = compute_hawkes(model, ev, window=influence_window)
        if lam is not None:
            start = max(0, len(ev["T"]) - len(lam))
            T_w = ev["T"][start:]
            st.plotly_chart(intensity_plot(T_w, lam, psi), use_container_width=True)

        # Cascade probability
        story_section("Cascade Probability", "psi/lambda: fraction of intensity from endogenous excitation.")
        if lam is not None and len(lam) > 0:
            prob = psi / np.maximum(lam, 1e-8)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=T_w, y=prob, mode="lines", name="psi/lambda"))
            fig.update_layout(height=300, xaxis_title="Time (hours)")
            st.plotly_chart(apply_fig_style(fig), use_container_width=True)

        # Real vs generated comparison
        story_section("Real vs Generated", "Side-by-side sphere and distribution comparison.")
        if sim is not None and d_assets == 3:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    sphere_scatter(real["W"][:n_events], real["R"][:n_events], title="Real", labels=symbols),
                    use_container_width=True,
                )
            with c2:
                st.plotly_chart(
                    sphere_scatter(sim["W"][:n_events], sim["R"][:n_events], title="Generated", labels=symbols),
                    use_container_width=True,
                )
            st.plotly_chart(radial_histogram(real["R"][:n_events], sim["R"][:n_events]), use_container_width=True)
        else:
            st.plotly_chart(radial_histogram(ev["R"]), use_container_width=True)

    # ── Analyst mode ──────────────────────────────────────────────────
    elif mode == "Analyst":
        story_section("Analyst Mode", "Diagnostics and model internals.")

        ev = subset_events(chosen, n_events)
        lam, psi, kernel = compute_hawkes(model, ev, window=influence_window)

        if d_assets == 3:
            color_choice = st.selectbox("Color sphere by", ["Magnitude (R)", "Cascade Prob (psi/lambda)"])
            if color_choice.startswith("Cascade") and lam is not None:
                prob = psi / np.maximum(lam, 1e-8)
                start = max(0, len(ev["T"]) - len(prob))
                # Pad to full length
                full_prob = np.zeros(len(ev["T"]))
                full_prob[start:] = prob
                st.plotly_chart(
                    sphere_scatter(ev["W"], full_prob, color_label="psi/lambda", title="Events colored by cascade probability", labels=symbols),
                    use_container_width=True,
                )
            else:
                st.plotly_chart(
                    sphere_scatter(ev["W"], ev["R"], color_label="R", title="Events colored by magnitude", labels=symbols),
                    use_container_width=True,
                )

        # Intensity
        if lam is not None:
            start = max(0, len(ev["T"]) - len(lam))
            T_w = ev["T"][start:]
            st.plotly_chart(intensity_plot(T_w, lam, psi), use_container_width=True)

        # Subcriticality
        if kernel is not None:
            st.plotly_chart(subcriticality_plot(kernel), use_container_width=True)

        # Attenuation heatmap
        if d_assets == 3:
            st.plotly_chart(attenuation_heatmap(ev["W"], model, labels=symbols), use_container_width=True)

        # Hawkes influence heatmap
        if kernel is not None:
            fig = go.Figure()
            fig.add_trace(go.Heatmap(z=kernel, colorscale="Viridis"))
            fig.update_layout(height=400, xaxis_title="Past event j", yaxis_title="Current event i", title="Hawkes Kernel Matrix")
            st.plotly_chart(apply_fig_style(fig), use_container_width=True)

        # Prompt generator
        st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
        story_section("Prompt Generator", "Simulate a cascade from a single shock.")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            prompt_asset = st.selectbox("Shock asset", symbols)
        with col_p2:
            prompt_mag = st.number_input("Magnitude (neg=crash, pos=rally)", value=-3.0, step=0.5)
        with col_p3:
            prompt_horizon = st.number_input("Horizon (hours)", value=200.0, step=50.0)

        if st.button("Generate cascade from prompt"):
            asset_idx = symbols.index(prompt_asset)
            with st.spinner("Running Ogata thinning..."):
                gen_ev, gen_gen = prompt_generate(
                    asset_idx, prompt_mag, prompt_horizon,
                    max_events=cfg["simulate"]["horizon_events"],
                    model=model, q_model=q_model,
                    safety_factor=cfg["simulate"].get("safety_factor", 1.5),
                )
            st.success(f"Generated {len(gen_ev['T'])} events, {gen_gen.immigrant_mask.sum()} immigrants")
            if d_assets == 3:
                st.plotly_chart(
                    sphere_scatter(gen_ev["W"], gen_ev["R"], title="Prompted Cascade", labels=symbols),
                    use_container_width=True,
                )
            st.plotly_chart(
                timeline_plot(gen_ev["T"], color_values=gen_gen.cascade_probs, title="Prompted Timeline"),
                use_container_width=True,
            )
            st.plotly_chart(
                genealogy_tree_plot(gen_ev["T"], gen_gen.parents, gen_gen.cascade_probs, title="Prompted Genealogy"),
                use_container_width=True,
            )

    # ── Genealogy mode ────────────────────────────────────────────────
    elif mode == "Genealogy":
        story_section("Genealogy Mode", "Immigration-branching structure of cascading extremes.")

        ev = subset_events(chosen, n_events)
        lam, psi, kernel = compute_hawkes(model, ev, window=min(influence_window, len(ev["T"])))

        if lam is None or kernel is None:
            st.warning("Not enough events for genealogy analysis.")
            return

        start = max(0, len(ev["T"]) - len(lam))

        # Build genealogy from windowed data
        genealogy = build_genealogy(lam, psi, kernel)
        n_imm = genealogy.immigrant_mask.sum()
        n_total = len(genealogy.parents)

        col1, col2, col3 = st.columns(3)
        col1.metric("Events (window)", str(n_total))
        col2.metric("Immigrants", f"{n_imm} ({100*n_imm/max(n_total,1):.1f}%)")
        col3.metric("Clusters", str(len(genealogy.clusters)))

        # Genealogy tree
        T_w = ev["T"][start:]
        st.plotly_chart(
            genealogy_tree_plot(T_w, genealogy.parents, genealogy.cascade_probs),
            use_container_width=True,
        )

        # Cascade probability distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=genealogy.cascade_probs, nbinsx=30, marker_color="#475569"))
        fig.update_layout(height=300, xaxis_title="psi/lambda", title="Cascade Probability Distribution")
        st.plotly_chart(apply_fig_style(fig), use_container_width=True)

        # Cluster explorer
        story_section("Cluster Explorer", "Select a cluster to inspect its event chain and parent links.")
        sorted_clusters = sorted(genealogy.clusters.keys())
        cluster_sizes = {k: len(v) for k, v in genealogy.clusters.items()}
        # Sort by size descending for exploration
        sorted_by_size = sorted(cluster_sizes.items(), key=lambda x: -x[1])
        cluster_options = [f"Cluster {k} ({sz} events)" for k, sz in sorted_by_size]

        if cluster_options:
            selected = st.selectbox("Select cluster", cluster_options)
            cluster_id = int(selected.split(" ")[1])
            cluster_indices = genealogy.clusters[cluster_id]

            W_w = ev["W"][start:]
            R_w = ev["R"][start:]

            if d_assets == 3:
                st.plotly_chart(
                    cluster_plot(T_w, W_w, R_w, cluster_indices, cluster_id, labels=symbols),
                    use_container_width=True,
                )

            # Show cluster details
            st.markdown(f"**Events in cluster {cluster_id}:**")
            for idx in cluster_indices:
                parent = genealogy.parents[idx]
                cp = genealogy.cascade_probs[idx]
                w_str = ", ".join(f"{w:.3f}" for w in W_w[idx])
                st.text(f"  [{idx}] T={T_w[idx]:.1f}h  R={R_w[idx]:.2f}  W=({w_str})  parent={parent}  cp={cp:.3f}")

        # Subcriticality check
        story_section("Subcriticality Check", "Kernel row sums must stay below the subcriticality bound.")
        st.plotly_chart(subcriticality_plot(kernel), use_container_width=True)


if __name__ == "__main__":
    main()
