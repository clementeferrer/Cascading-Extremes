"""CSS and layout helpers — same visual language as app_phase2.py."""

import textwrap
import streamlit as st


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
        /* ── Print ── */
        @media print {
            /* KEY FIX: override flexbox → block so break-inside works */
            div[data-testid="stVerticalBlock"],
            div[data-testid="stHorizontalBlock"],
            div[data-testid="stVerticalBlockBorderWrapper"],
            .stMainBlockContainer > div,
            section[data-testid="stMain"] > div,
            div[data-testid="stColumn"] > div {
                display: block !important;
            }

            /* Now break-inside is respected */
            div[data-testid="stPlotlyChart"],
            div[data-testid="stMetric"],
            .section-block, .hero, .panel {
                break-inside: avoid !important;
                page-break-inside: avoid !important;
            }

            /* Hide interactive-only elements */
            [data-testid="stSidebar"],
            .stButton,
            div[data-testid="stSlider"],
            header[data-testid="stHeader"],
            footer {
                display: none !important;
            }

            /* White background */
            .stApp {
                background: white !important;
            }

            /* Avoid orphan headings */
            h1, h2, h3, h4 {
                break-after: avoid !important;
                page-break-after: avoid !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero_section(title: str, subtitle: str, tag: str):
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


def story_section(title: str, subtitle: str):
    html = f"""
    <div class="section-block">
        <div class="section-headline">{title}</div>
        <div class="section-sub">{subtitle}</div>
    </div>
    """
    st.markdown(textwrap.dedent(html), unsafe_allow_html=True)


def panel(title: str, body: str):
    st.markdown(
        f'<div class="panel"><h4>{title}</h4><p>{body}</p></div>',
        unsafe_allow_html=True,
    )
