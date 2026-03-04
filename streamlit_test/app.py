"""Streamlit Cascade Explorer — Interactive 3D viewer, theory, and experimentation.

Run from project root:
    streamlit run streamlit_test/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path so second_phase/ and cascades/ are importable
ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st

from streamlit_test.styles import inject_styles
from streamlit_test.data_loader import load_all_data, load_global_data, load_var_data


def main():
    st.set_page_config(
        page_title="Cascade Explorer",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    # ── Sidebar navigation ───────────────────────────────────────────
    st.sidebar.title("Cascade Explorer")

    source = st.sidebar.radio(
        "Data Source",
        ["Real Data", "Real Data (Global u_\u03c4)", "Simulated (VAR)"],
    )

    page = st.sidebar.radio(
        "Navigate",
        ["3D Viewer", "Theory & Genealogy", "Experimentation"],
    )

    # ── Load data ─────────────────────────────────────────────────────
    try:
        if source == "Simulated (VAR)":
            data = load_var_data()
        elif source == "Real Data (Global u_\u03c4)":
            data = load_global_data()
        else:
            data = load_all_data()
    except FileNotFoundError as e:
        if source == "Simulated (VAR)":
            st.error(
                f"VAR artifacts not found: {e}\n\n"
                "Run the training pipeline first:\n"
                "```\npython -m streamlit_test.train_var --config configs/phase2_var.yaml\n```"
            )
        elif source == "Real Data (Global u_\u03c4)":
            st.error(
                f"Global threshold artifacts not found: {e}\n\n"
                "Train the global model first:\n"
                "```\npython -m second_phase.train --config configs/phase2_global.yaml\n"
                "python -m second_phase.simulate --config configs/phase2_global.yaml\n```"
            )
        else:
            st.error(f"Data not found: {e}. Ensure artifacts/phase2/ and data/processed_phase2/ exist.")
        return
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    # ── Sidebar info ──────────────────────────────────────────────────
    st.sidebar.markdown("---")
    threshold_label = "global" if data.threshold_mode == "global" else "directional"
    st.sidebar.caption(
        f"**Source:** {source}  \n"
        f"**Assets:** {', '.join(data.symbols)}  \n"
        f"**Events:** {len(data.real_events['T'])}  \n"
        f"**tau:** {data.tau}  \n"
        f"**Threshold:** {threshold_label}"
    )

    # ── Dispatch to page ─────────────────────────────────────────────
    if page == "3D Viewer":
        from streamlit_test.pages_viewer import render_viewer
        render_viewer(data)
    elif page == "Theory & Genealogy":
        from streamlit_test.pages_theory import render_theory
        render_theory(data)
    elif page == "Experimentation":
        from streamlit_test.pages_experiment import render_experiment
        render_experiment(data)


if __name__ == "__main__":
    main()
