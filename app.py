"""Thin entry point for Streamlit Cloud — delegates to streamlit_test.app."""
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from streamlit_test.app import main

main()
