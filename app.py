"""Entry point:  streamlit run app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

st.set_page_config(page_title="Explainable Fraud Detection", page_icon="🛡️", layout="wide")

pages = [
    st.Page("views/data.py", title="1 · Data", icon="📥"),
    st.Page("views/preparation.py", title="2 · Preparation", icon="🧪"),
    st.Page("views/modeling.py", title="3 · Models & tuning", icon="🤖"),
    st.Page("views/evaluation.py", title="4 · Evaluation & threshold", icon="📊"),
    st.Page("views/explain.py", title="5 · Explainability", icon="🔍"),
    st.Page("views/export.py", title="6 · Export & API", icon="🚀"),
]
nav = st.navigation(pages)

from ui.state import S  # noqa: E402

with st.sidebar:
    s = S()
    st.caption("Pipeline status")
    st.write(("✅" if s.df is not None else "⬜") + " Data loaded")
    st.write(("✅" if s.prepared is not None else "⬜") + " Prepared")
    st.write(("✅" if s.models else "⬜") + f" Models trained ({len(s.models)})")
nav.run()
