"""Reusable widgets."""
from __future__ import annotations

import streamlit as st

from fraud_xai.models.registry import ModelSpec


def param_form(spec: ModelSpec, prefix: str) -> dict:
    """Render one widget per hyper-parameter declared in the registry."""
    vals = {}
    cols = st.columns(2)
    for i, p in enumerate(spec.params):
        k, box = f"{prefix}_{spec.key}_{p.name}", cols[i % 2]
        with box:
            if p.kind == "int":
                vals[p.name] = st.slider(p.name, int(p.low), int(p.high), int(p.default), key=k, help=p.help or None)
            elif p.kind == "float" and p.log:
                vals[p.name] = st.number_input(p.name, float(p.low), float(p.high), float(p.default),
                                               format="%.4g", key=k, help=p.help or None)
            elif p.kind == "float":
                vals[p.name] = st.slider(p.name, float(p.low), float(p.high), float(p.default),
                                         key=k, help=p.help or None)
            elif p.kind == "choice":
                vals[p.name] = st.selectbox(p.name, list(p.choices), index=list(p.choices).index(p.default),
                                            key=k, help=p.help or None)
            else:
                vals[p.name] = st.checkbox(p.name, bool(p.default), key=k, help=p.help or None)
    return vals


def fmt_money(x: float) -> str:
    return f"{x:,.0f}"
