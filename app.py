from pathlib import Path
from datetime import datetime
import textwrap
import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from cascades.dataset import zeta
from cascades.model import CascadingTransformer, ModelConfig
from cascades.utils import load_config, ensure_dir


@st.cache_resource
def load_model():
    payload = torch.load("artifacts/model.pt", map_location="cpu")
    cfg = ModelConfig(**payload["model_cfg"])
    model = CascadingTransformer(d_input=payload["d_input"], d_assets=payload["d_assets"], cfg=cfg)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def inject_styles():
    st.markdown(
        """
        <style>
        :root { color-scheme: light; }
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Newsreader:wght@400;600&display=swap');
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
        [data-testid="stSidebar"] .stButton button {
            background: #0f172a;
            color: #ffffff;
            border: 1px solid #0f172a;
        }
        [data-testid="stSidebar"] .stButton button:hover {
            background: #111827;
            color: #ffffff;
        }
        [data-testid="stSidebar"] div[data-baseweb="select"] > div {
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
        h1, h2, h3, h4 {
            font-family: 'Newsreader', serif;
            letter-spacing: 0.2px;
        }
        .hero {
            background: linear-gradient(135deg, #0f172a 0%, #1f2a44 100%);
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
        .section-title {
            font-size: 20px;
            font-weight: 700;
            margin: 16px 0 8px 0;
        }
        .section-block {
            background: #ffffff;
            border-radius: 16px;
            padding: 18px 20px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
            margin: 10px 0 12px 0;
        }
        .section-kicker {
            font-size: 11px;
            letter-spacing: 1.6px;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 6px;
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
        .sidebar-note {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 10px 12px;
            font-size: 12px;
            color: #0f172a;
            box-shadow: 0 6px 14px rgba(15, 23, 42, 0.06);
            margin-bottom: 10px;
        }
        .metric-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 14px 16px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
        }
        .metric-title {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #475569;
            margin-bottom: 6px;
        }
        .metric-value {
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 4px;
            color: #0f172a;
        }
        .metric-sub {
            font-size: 13px;
            color: #0f172a;
        }
        .metric-card .katex, .metric-card .katex * {
            color: #0f172a !important;
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


def story_section(title, subtitle, step=None):
    kicker = ""
    kicker_html = f'<div class="section-kicker">{kicker}</div>' if kicker else ""
    html = f"""
    <div class="section-block">
        {kicker_html}
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
    a_text = None
    b_text = None
    c_text = None
    if fig.layout.ternary:
        if fig.layout.ternary.aaxis and fig.layout.ternary.aaxis.title:
            a_text = fig.layout.ternary.aaxis.title.text
        if fig.layout.ternary.baxis and fig.layout.ternary.baxis.title:
            b_text = fig.layout.ternary.baxis.title.text
        if fig.layout.ternary.caxis and fig.layout.ternary.caxis.title:
            c_text = fig.layout.ternary.caxis.title.text

    fig.update_layout(
        ternary=dict(
            aaxis=dict(
                title=dict(font=dict(color="#0f172a"), text=a_text) if a_text else dict(font=dict(color="#0f172a")),
                tickfont=dict(color="#0f172a"),
                color="#0f172a",
            ),
            baxis=dict(
                title=dict(font=dict(color="#0f172a"), text=b_text) if b_text else dict(font=dict(color="#0f172a")),
                tickfont=dict(color="#0f172a"),
                color="#0f172a",
            ),
            caxis=dict(
                title=dict(font=dict(color="#0f172a"), text=c_text) if c_text else dict(font=dict(color="#0f172a")),
                tickfont=dict(color="#0f172a"),
                color="#0f172a",
            ),
        )
    )
    return fig


def subset_events(events, n_events):
    return {k: v[:n_events] for k, v in events.items()}


def spotlight_indices(scores, top_k=5):
    if scores is None or len(scores) == 0:
        return np.array([], dtype=int)
    top_k = max(1, min(top_k, len(scores)))
    return np.argsort(scores)[-top_k:]


def compute_tokens(W, R, dT):
    z = zeta(W)
    tokens = np.concatenate([W, z, np.log(R + 1.0e-8)[:, None], np.log(dT + 1.0e-8)[:, None]], axis=1)
    return tokens


def ensure_events(events):
    if "tokens" not in events:
        events["tokens"] = compute_tokens(events["W"], events["R"], events["dT"])
    if "u" not in events:
        events["u"] = np.ones_like(events["R"])
    return events


def cascade_probabilities(model, events, window=256):
    events = ensure_events(events)
    W = events["W"]
    R = events["R"]
    dT = events["dT"]
    T = events["T"]
    u = events["u"]
    tokens = events["tokens"]

    if len(tokens) < 3:
        return np.array([]), np.array([]), np.array([])

    max_len = getattr(model.cfg, "max_len", window)
    window = min(window, max_len)
    start = max(0, len(tokens) - window)
    tokens = tokens[start:]
    W = W[start:]
    R = R[start:]
    dT = dT[start:]
    T = T[start:]
    u = u[start:]

    tokens_in = torch.tensor(tokens[:-1][None, :, :], dtype=torch.float32)
    W_next = torch.tensor(W[1:][None, :, :], dtype=torch.float32)
    R_next = torch.tensor(R[1:][None, :], dtype=torch.float32)
    u_next = torch.tensor(u[1:][None, :], dtype=torch.float32)
    dT_next = torch.tensor(dT[1:][None, :], dtype=torch.float32)
    T_in = torch.tensor(T[:-1][None, :], dtype=torch.float32)
    R_in = torch.tensor(R[:-1][None, :], dtype=torch.float32)

    with torch.no_grad():
        out = model.log_likelihood(tokens_in, W_next, R_next, dT_next, T_in, R_in, u_next)
        psi = out["psi"].squeeze(0).numpy()
        lam = out["lambda"].squeeze(0).numpy()
        prob = psi / np.maximum(lam, 1.0e-8)
    return prob, lam, psi, T[1:]


def hawkes_influence(model, events, window=128):
    events = ensure_events(events)
    tokens = events["tokens"]
    R = events["R"]
    T = events["T"]

    if len(tokens) < 3:
        return None, None, None

    max_len = getattr(model.cfg, "max_len", window)
    window = min(window, max_len)
    start = max(0, len(tokens) - window)
    tokens = tokens[start:]
    R = R[start:]
    T = T[start:]

    tokens_t = torch.tensor(tokens[None, :, :], dtype=torch.float32)
    T_tensor = torch.tensor(T[None, :], dtype=torch.float32)
    R_tensor = torch.tensor(R[None, :], dtype=torch.float32)
    log_r = torch.log(R_tensor + 1.0e-8)

    with torch.no_grad():
        h = model.encode(tokens_t)
        try:
            lam, psi, kernel = model.hawkes_intensity(h, T_tensor, log_r, return_kernel=True)
        except TypeError:
            lam, psi = model.hawkes_intensity(h, T_tensor, log_r)
            kernel = None

    kernel_np = kernel.squeeze(0).numpy() if kernel is not None else None
    return lam.squeeze(0).numpy(), psi.squeeze(0).numpy(), kernel_np


def narrator_panel(symbols, events, prob, lam, psi):
    w_mean = events["W"].mean(axis=0)
    dominant_idx = int(np.argmax(w_mean))
    dominant_asset = symbols[dominant_idx] if dominant_idx < len(symbols) else f"Asset {dominant_idx + 1}"
    dominant_share = float(w_mean[dominant_idx])

    avg_dt = float(np.mean(events["dT"])) if len(events["dT"]) > 0 else 0.0
    event_rate = 1.0 / avg_dt if avg_dt > 0 else 0.0
    median_r = float(np.median(events["R"])) if len(events["R"]) > 0 else 0.0
    tail_ratio = float(np.quantile(events["R"], 0.95) / (median_r + 1.0e-8)) if len(events["R"]) > 0 else 0.0
    directional_weight = (events["W"] * events["R"][:, None]).mean(axis=0)
    dominant_mag_idx = int(np.argmax(directional_weight))
    dominant_mag_asset = symbols[dominant_mag_idx] if dominant_mag_idx < len(symbols) else f"Asset {dominant_mag_idx + 1}"

    p_mean = float(np.mean(prob)) if len(prob) > 0 else 0.0
    p_max = float(np.max(prob)) if len(prob) > 0 else 0.0

    st.markdown("**Live Paper Narrator**")
    col1, col2, col3 = st.columns(3)
    col1.markdown(
        f"""
        <div class="panel">
            <h4>Finance and Trading</h4>
            <p>Dominant extreme direction: {dominant_asset} with mean share {dominant_share:.2f}. Event rate is {event_rate:.2f} per hour in this window.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"""
        <div class="panel">
            <h4>Statistics</h4>
            <p>Median radial magnitude is {median_r:.2f}. The 95th percentile is {tail_ratio:.1f} times larger. Highest magnitudes concentrate toward {dominant_mag_asset}.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col3.markdown(
        f"""
        <div class="panel">
            <h4>ML and AI</h4>
            <p>Average cascade probability is {p_mean:.3f} with a peak of {p_max:.3f}, showing episodic endogenous clustering learned by the transformer.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def export_story_slides(figs, out_dir="artifacts/story_slides"):
    ensure_dir(out_dir)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    exported = []
    for name, fig in figs.items():
        if fig is None:
            continue
        fname = f"{ts}_{name}.png"
        out_path = Path(out_dir) / fname
        try:
            fig.write_image(out_path, scale=2)
            exported.append(str(out_path))
        except Exception as exc:
            st.error("Export failed. Install kaleido with: pip install kaleido")
            st.error(str(exc))
            return []
    return exported


def timeline_plot(real_T, sim_T):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=real_T, y=np.zeros_like(real_T), mode="markers", name="Real", marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=sim_T, y=np.ones_like(sim_T), mode="markers", name="Generated", marker=dict(size=4)))
    fig.update_layout(height=300, yaxis=dict(showticklabels=False, title=""), xaxis_title="Time (hours)")
    return apply_fig_style(fig)


def timeline_plot_single(T, spotlight_idx=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=T, y=np.zeros_like(T), mode="markers", name="Events", marker=dict(size=4, color="#475569")))
    if spotlight_idx is not None and len(spotlight_idx) > 0:
        fig.add_trace(
            go.Scatter(
                x=T[spotlight_idx],
                y=np.zeros_like(spotlight_idx),
                mode="markers",
                name="Spotlight",
                marker=dict(size=9, color="#e76f51"),
            )
        )
    fig.update_layout(height=260, yaxis=dict(showticklabels=False, title=""), xaxis_title="Time (hours)")
    return apply_fig_style(fig)


def ternary_plot_single(W, spotlight_idx=None, labels=None):
    fig = go.Figure()
    fig.add_trace(
        go.Scatterternary(
            a=W[:, 0],
            b=W[:, 1],
            c=W[:, 2],
            mode="markers",
            name="Events",
            marker=dict(size=5, color="#475569", opacity=0.7),
        )
    )
    if spotlight_idx is not None and len(spotlight_idx) > 0:
        fig.add_trace(
            go.Scatterternary(
                a=W[spotlight_idx, 0],
                b=W[spotlight_idx, 1],
                c=W[spotlight_idx, 2],
                mode="markers",
                name="Spotlight",
                marker=dict(size=9, color="#e76f51", opacity=0.9),
            )
        )
    if labels and len(labels) == 3:
        fig.update_layout(
            ternary=dict(
                aaxis=dict(title=dict(text=labels[0])),
                baxis=dict(title=dict(text=labels[1])),
                caxis=dict(title=dict(text=labels[2])),
            )
        )
    fig.update_layout(height=360)
    return apply_fig_style(fig)


def ternary_plot(W_real, W_sim, labels=None):
    fig = go.Figure()
    fig.add_trace(go.Scatterternary(
        a=W_real[:, 0], b=W_real[:, 1], c=W_real[:, 2],
        mode="markers", name="Real", marker=dict(size=4, opacity=0.6)
    ))
    fig.add_trace(go.Scatterternary(
        a=W_sim[:, 0], b=W_sim[:, 1], c=W_sim[:, 2],
        mode="markers", name="Generated", marker=dict(size=4, opacity=0.6)
    ))
    if labels and len(labels) == 3:
        fig.update_layout(
            ternary=dict(
                aaxis=dict(title=dict(text=labels[0])),
                baxis=dict(title=dict(text=labels[1])),
                caxis=dict(title=dict(text=labels[2])),
            )
        )
    fig.update_layout(height=400)
    return apply_fig_style(fig)


def radial_plot(R_real, R_sim):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=R_real, name="Real", opacity=0.6))
    fig.add_trace(go.Histogram(x=R_sim, name="Generated", opacity=0.6))
    fig.update_layout(barmode="overlay", height=300)
    return apply_fig_style(fig)


def angle_magnitude_ternary(W, R, title="Angle-Dependent Magnitude", labels=None):
    fig = go.Figure()
    fig.add_trace(
        go.Scatterternary(
            a=W[:, 0],
            b=W[:, 1],
            c=W[:, 2],
            mode="markers",
            marker=dict(
                size=6,
                color=R,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=dict(text="R", font=dict(color="#0f172a")), tickfont=dict(color="#0f172a")),
                opacity=0.8,
            ),
            name="Magnitude",
        )
    )
    if labels and len(labels) == 3:
        fig.update_layout(
            ternary=dict(
                aaxis=dict(title=dict(text=labels[0])),
                baxis=dict(title=dict(text=labels[1])),
                caxis=dict(title=dict(text=labels[2])),
            )
        )
    fig = apply_fig_style(fig)
    fig.update_layout(title=title, height=450, margin=dict(l=20, r=20, t=40, b=80))
    return fig


def _animation_controls(fig):
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.98,
                "y": -0.15,
                "xanchor": "right",
                "yanchor": "top",
                "pad": {"r": 10, "t": 0},
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ]
    )
    return fig


def animate_timeline(T, step=5, title="Cascade Timeline", spotlight_idx=None):
    n = len(T)
    step = max(1, step)
    frames = []
    spotlight_idx = np.array(spotlight_idx) if spotlight_idx is not None else np.array([], dtype=int)
    for k in range(1, n + 1, step):
        base = go.Scatter(x=T[:k], y=np.zeros(k), mode="markers", marker=dict(size=5, color="#475569"))
        if len(spotlight_idx) > 0:
            mask = spotlight_idx[spotlight_idx < k]
            highlight = go.Scatter(
                x=T[mask],
                y=np.zeros_like(mask),
                mode="markers",
                marker=dict(size=9, color="#e76f51"),
                name="Spotlight",
            )
            frames.append(go.Frame(data=[base, highlight], name=str(k)))
        else:
            frames.append(go.Frame(data=[base], name=str(k)))

    fig = go.Figure(
        data=[go.Scatter(x=T[:1], y=np.zeros(1), mode="markers", marker=dict(size=6, color="#475569"))],
        frames=frames,
    )
    fig.update_layout(title=title, height=280, yaxis=dict(showticklabels=False, title=""), xaxis_title="Time (hours)")
    return _animation_controls(apply_fig_style(fig))


def animate_ternary(W, step=5, title="Cascade Geometry (W)", spotlight_idx=None, labels=None, color_values=None):
    if W.shape[1] != 3:
        fig = go.Figure()
        fig.add_annotation(text="Ternary plot requires exactly 3 assets.")
        return fig

    n = len(W)
    step = max(1, step)
    frames = []
    spotlight_idx = np.array(spotlight_idx) if spotlight_idx is not None else np.array([], dtype=int)
    for k in range(1, n + 1, step):
        color_slice = color_values[:k] if color_values is not None else None
        base_marker = dict(size=5, opacity=0.7)
        if color_slice is not None:
            base_marker.update(dict(color=color_slice, colorscale="Viridis", showscale=False))
        else:
            base_marker.update(dict(color="#475569"))
        base = go.Scatterternary(
            a=W[:k, 0],
            b=W[:k, 1],
            c=W[:k, 2],
            mode="markers",
            marker=base_marker,
        )
        if len(spotlight_idx) > 0:
            mask = spotlight_idx[spotlight_idx < k]
            highlight = go.Scatterternary(
                a=W[mask, 0],
                b=W[mask, 1],
                c=W[mask, 2],
                mode="markers",
                marker=dict(size=9, color="#e76f51", opacity=0.9),
                name="Spotlight",
            )
            frames.append(go.Frame(data=[base, highlight], name=str(k)))
        else:
            frames.append(go.Frame(data=[base], name=str(k)))

    init_marker = dict(size=6)
    if color_values is not None:
        init_marker.update(dict(color=[color_values[0]], colorscale="Viridis", showscale=False))
    else:
        init_marker.update(dict(color="#475569"))
    fig = go.Figure(
        data=[go.Scatterternary(a=[W[0, 0]], b=[W[0, 1]], c=[W[0, 2]], mode="markers", marker=init_marker)],
        frames=frames,
    )
    if labels and len(labels) == 3:
        fig.update_layout(
            ternary=dict(
                aaxis=dict(title=dict(text=labels[0])),
                baxis=dict(title=dict(text=labels[1])),
                caxis=dict(title=dict(text=labels[2])),
            )
        )
    fig.update_layout(title=title, height=400)
    return _animation_controls(apply_fig_style(fig))


def influence_heatmap(kernel, title="Hawkes Influence Map"):
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=kernel, colorscale="Viridis"))
    fig.update_layout(title=title, height=400, xaxis_title="Past event j", yaxis_title="Current event i")
    return apply_fig_style(fig)


def intensity_plot(T, lam, psi):
    mu = lam - psi
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=T, y=lam, mode="lines", name="lambda", line=dict(color="#1f2937", width=3)))
    fig.add_trace(go.Scatter(x=T, y=mu, mode="lines", name="mu (exogenous)", line=dict(color="#94a3b8", width=2, dash="dash")))
    fig.add_trace(
        go.Scatter(
            x=T,
            y=psi,
            mode="lines",
            name="psi (endogenous)",
            line=dict(color="#e76f51", width=3),
            fill="tozeroy",
            fillcolor="rgba(231,111,81,0.15)",
        )
    )
    fig.update_layout(height=300, xaxis_title="Time (hours)")
    return apply_fig_style(fig)




def cascade_3d_plot(W, R, T, title="Cascading Extremes on the Simplex (3D)", labels=None):
    x_label = "W1"
    y_label = "W2"
    z_label = "W3"
    if labels and len(labels) == 3:
        x_label, y_label, z_label = labels
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=W[:, 0],
            y=W[:, 1],
            z=W[:, 2],
            mode="markers",
            marker=dict(size=4, color=R, colorscale="Viridis", opacity=0.85, colorbar=dict(title=dict(text="R"))),
            name="Extremes",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=W[:, 0],
            y=W[:, 1],
            z=W[:, 2],
            mode="lines",
            line=dict(color="rgba(15,23,42,0.25)", width=3),
            name="Temporal path",
        )
    )
    fig.update_layout(
        height=480,
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label,
            bgcolor="rgba(255,255,255,0.0)",
        ),
        title=title,
    )
    return apply_fig_style(fig)


def main():
    st.set_page_config(page_title="Cascading Extremes", layout="wide")
    inject_styles()
    hero_section(
        "Live Paper: Cascading Extremes",
        "Generative cascades of multivariate extremes with causal attention",
        "EVT x Transformer",
    )

    if not Path("artifacts/model.pt").exists():
        st.warning("Model not found. Train first with `python -m cascades.train --config configs/default.yaml`.")
        return

    cfg = load_config("configs/default.yaml")
    model = load_model()
    real = ensure_events(dict(np.load("data/processed/events.npz")))
    sim_path = Path("artifacts/simulated_events.npz")
    sim = None
    if sim_path.exists():
        sim = ensure_events(dict(np.load(sim_path)))

    symbols = None
    meta_path = Path("artifacts/meta.json")
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                symbols = json.load(f).get("symbols")
        except Exception:
            symbols = None
    if symbols is None:
        symbols = cfg["data"]["symbols"]

    st.sidebar.header("Director Controls")
    st.sidebar.markdown(
        """
        <div class="sidebar-note">
            Select the narrative mode and the event window. Use Spotlight to surface the most influential
            events. The goal is a cinematic, evidence-first story of how cascades form.
        </div>
        """,
        unsafe_allow_html=True,
    )
    mode = st.sidebar.selectbox("Mode", ["Story", "Analyst"])
    n_events = st.sidebar.slider("Max events", min_value=100, max_value=2000, value=500, step=50)
    anim_step = st.sidebar.slider("Animation step", min_value=1, max_value=20, value=5)
    influence_window = st.sidebar.slider("Influence window", min_value=20, max_value=200, value=120)
    dataset_choice = st.sidebar.selectbox("Cascade for animation", ["Real", "Generated"])
    spotlight_on = st.sidebar.checkbox("Spotlight causal events (top psi)", value=True)
    st.sidebar.caption(r"Highlights top-$\psi$ events in orange on the timeline and simplex.")
    spotlight_k = st.sidebar.slider("Spotlight count", min_value=3, max_value=15, value=7, step=1)
    export_slides = st.sidebar.checkbox("Enable slide export", value=False)

    if sim is None:
        st.warning("Simulated events not found. Run `python -m cascades.simulate --config configs/default.yaml` to enable generated views.")

    chosen = real if dataset_choice == "Real" or sim is None else sim

    if mode == "Story":
        story_section(
            "Story Mode: How Cascades Form",
            "An evidence-first narrative for Finance, Statistics, and ML/AI. Each panel is a stage in the cascade.",
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Assets", ", ".join(symbols))
        tau_val = f"{cfg['extremes']['tau']:.2f}"
        col2.markdown('<div class="metric-card">', unsafe_allow_html=True)
        col2.markdown('<div class="metric-title">Directional quantile</div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-value">{tau_val}</div>', unsafe_allow_html=True)
        col2.latex(r"\tau \text{ in } u_{\tau}(\mathbf{w}) \text{ for } R \mid \mathbf{W}=\mathbf{w}")
        col2.markdown('</div>', unsafe_allow_html=True)
        col3.metric("Total events", f"{len(chosen['T'])}")
        col4.metric("Window size", f"{n_events}")

        st.markdown(
            "This view narrates how a cascade builds in time: timing, geometry, and causal pressure."
        )
        st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)
        col_a.markdown(
            """
            <div class="panel">
                <h4>Research Framing</h4>
                <p>Extremes are treated as a marked point process with causal attention. The transformer learns latent history that governs direction, magnitude, and timing.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_b.markdown(
            """
            <div class="panel">
                <h4>How to Read the Story</h4>
                <p>Start with the timeline, then observe geometric drift on the simplex. Finally, compare endogenous pressure to the baseline intensity.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_c.markdown(
            """
            <div class="panel">
                <h4>Key Signal</h4>
                <p>When the endogenous term dominates, the cascade is self-exciting. This is the signature of learned extreme dynamics.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        story_section(
            "Research Brief",
            "Data, model, and interpretation stitched into a single narrative.",
        )
        brief_a, brief_b, brief_c = st.columns(3)
        brief_a.markdown(
            """
            <div class="panel">
                <h4>Data</h4>
                <p>Hourly BTC/ETH/BNB returns. GARCH filtering and PIT enforce standard exponential margins.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        brief_b.markdown(
            """
            <div class="panel">
                <h4>Model</h4>
                <p>Causal transformer with Dirichlet direction, truncated Gamma magnitude, and Hawkes time.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        brief_c.markdown(
            """
            <div class="panel">
                <h4>Interpretation</h4>
                <p>Endogenous spikes indicate cascades. Directional thresholds highlight geometric extremes.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

        story_events = subset_events(chosen, n_events)
        window = min(influence_window, len(story_events["T"]))
        lam, psi, kernel = hawkes_influence(model, story_events, window=window)
        start_idx = len(story_events["T"]) - window if window > 0 else 0
        spot_idx = spotlight_indices(psi, top_k=spotlight_k) + start_idx if spotlight_on and psi is not None else np.array([], dtype=int)

        story_section(
            "Timeline: When extremes occur",
            "Cascades appear as clusters in time. Dense bursts indicate self-excitation.",
        )
        T = story_events["T"]
        fig_timeline_anim = animate_timeline(T, step=anim_step, title=f"{dataset_choice} Cascade Timeline", spotlight_idx=spot_idx)
        fig_timeline_static = timeline_plot_single(T, spotlight_idx=spot_idx)
        st.plotly_chart(fig_timeline_anim, use_container_width=True)
        if spotlight_on:
            st.caption("Spotlighted events (top psi) are marked in orange.")
            st.plotly_chart(fig_timeline_static, use_container_width=True)

        story_section(
            "Geometry: Where extremes point",
            "Direction lives on the simplex. Drift or clustering signals structural dependence.",
        )
        st.markdown(r"Geometry on the simplex: $\boldsymbol{W}_t$")
        W = story_events["W"]
        fig_ternary_anim = animate_ternary(
            W,
            step=anim_step,
            title=f"{dataset_choice} Cascade Geometry (W)",
            spotlight_idx=spot_idx,
            labels=symbols,
            color_values=story_events["R"],
        )
        fig_ternary_static = ternary_plot_single(W, spotlight_idx=spot_idx, labels=symbols)
        st.plotly_chart(fig_ternary_anim, use_container_width=True)
        fig_angle = None
        if W.shape[1] == 3:
            fig_angle = angle_magnitude_ternary(
                W,
                story_events["R"],
                title="Angle-Dependent Magnitude (R over W)",
                labels=symbols,
            )
            st.plotly_chart(fig_angle, use_container_width=True)
            fig_3d = cascade_3d_plot(W, story_events["R"], story_events["T"], labels=symbols)
            st.plotly_chart(fig_3d, use_container_width=True)
            st.caption("Color encodes magnitude R; the line shows temporal order.")
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        story_section(
            "Causal Pressure: Endogenous vs Exogenous",
            "Endogenous spikes show the cascade feeding on itself. Exogenous intensity is the baseline.",
        )
        if lam is not None:
            T_window = story_events["T"][start_idx:]
            fig_intensity = intensity_plot(T_window, lam, psi)
            st.plotly_chart(fig_intensity, use_container_width=True)
            fig_heat = None
            if kernel is not None:
                fig_heat = influence_heatmap(kernel, title="Hawkes Influence Map (Recent Window)")
                st.plotly_chart(fig_heat, use_container_width=True)
        else:
            fig_intensity = None
            fig_heat = None

        story_section(
            "Cascade Probability",
            "Probability that each event is endogenous. Peaks identify cascade regimes.",
        )
        st.markdown(r"Endogenous probability: $\Psi / \Lambda$")
        prob, _, _, t_prob = cascade_probabilities(model, story_events, window=len(story_events["T"]))
        fig_prob = None
        if len(prob) > 0:
            fig_prob = go.Figure()
            fig_prob.add_trace(go.Scatter(x=t_prob, y=prob, mode="lines", name="Psi/Lambda"))
            fig_prob.update_layout(height=300, xaxis_title="Time (hours since start)")
            fig_prob = apply_fig_style(fig_prob)
            st.plotly_chart(fig_prob, use_container_width=True)

        story_section(
            "Real vs Generated: Is the story preserved?",
            "We compare geometry and magnitude to verify the learned cascade is faithful.",
        )
        fig_radial = None
        if sim is not None:
            st.plotly_chart(timeline_plot(real["T"][:n_events], sim["T"][:n_events]), use_container_width=True)
            st.plotly_chart(ternary_plot(real["W"][:n_events], sim["W"][:n_events], labels=symbols), use_container_width=True)
            fig_radial = radial_plot(real["R"][:n_events], sim["R"][:n_events])
            st.plotly_chart(fig_radial, use_container_width=True)
        else:
            fig_radial = radial_plot(real["R"][:n_events], real["R"][:n_events])
            st.plotly_chart(fig_radial, use_container_width=True)

        if export_slides:
            st.markdown('<div class="section-title">Export Live Paper Slides</div>', unsafe_allow_html=True)
            if st.button("Export story slides to artifacts/story_slides"):
                figs = {
                    "timeline_static": fig_timeline_static,
                    "geometry_static": fig_ternary_static,
                    "angle_magnitude": fig_angle,
                    "intensity": fig_intensity,
                    "heatmap": fig_heat,
                    "probability": fig_prob,
                    "radial": fig_radial,
                }
                exported = export_story_slides(figs)
                if exported:
                    st.success(f"Exported {len(exported)} slides to artifacts/story_slides")
    else:
        st.markdown('<div class="section-title">Comparative View</div>', unsafe_allow_html=True)
        if sim is not None:
            st.plotly_chart(timeline_plot(real["T"][:n_events], sim["T"][:n_events]), use_container_width=True)
            st.plotly_chart(ternary_plot(real["W"][:n_events], sim["W"][:n_events], labels=symbols), use_container_width=True)
            if real["W"].shape[1] == 3:
                st.plotly_chart(
                    angle_magnitude_ternary(real["W"][:n_events], real["R"][:n_events], labels=symbols),
                    use_container_width=True,
                )
            st.plotly_chart(radial_plot(real["R"][:n_events], sim["R"][:n_events]), use_container_width=True)
        else:
            if real["W"].shape[1] == 3:
                st.plotly_chart(
                    angle_magnitude_ternary(real["W"][:n_events], real["R"][:n_events], labels=symbols),
                    use_container_width=True,
                )
            st.plotly_chart(radial_plot(real["R"][:n_events], real["R"][:n_events]), use_container_width=True)

        st.markdown('<div class="section-title">Cascade Movie</div>', unsafe_allow_html=True)
        T = chosen["T"][:n_events]
        W = chosen["W"][:n_events]
        st.plotly_chart(animate_timeline(T, step=anim_step, title=f"{dataset_choice} Cascade Timeline"), use_container_width=True)
        st.plotly_chart(
            animate_ternary(W, step=anim_step, title=f"{dataset_choice} Cascade Geometry (W)", labels=symbols, color_values=chosen["R"][:n_events]),
            use_container_width=True,
        )

        st.markdown('<div class="section-title">Endogenous vs Exogenous Intensity</div>', unsafe_allow_html=True)
        lam, psi, kernel = hawkes_influence(model, chosen, window=influence_window)
        if lam is not None:
            start = max(0, len(chosen["T"]) - influence_window)
            T_window = chosen["T"][start:]
            st.plotly_chart(intensity_plot(T_window, lam, psi), use_container_width=True)
            if kernel is not None:
                st.plotly_chart(influence_heatmap(kernel, title="Hawkes Influence Map (Recent Window)"), use_container_width=True)

        st.markdown('<div class="section-title">Cascade Probability</div>', unsafe_allow_html=True)
        st.markdown(r"Endogenous probability: $\Psi / \Lambda$")
        prob, _, _, t_prob = cascade_probabilities(model, chosen, window=cfg["model"]["max_len"])
        if len(prob) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_prob, y=prob, mode="lines", name="Psi/Lambda"))
            fig.update_layout(height=300, xaxis_title="Time (hours since start)")
            fig = apply_fig_style(fig)
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
