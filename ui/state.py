"""Typed session state shared by all Streamlit pages."""
from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st

from fraud_xai.evaluation.threshold import CostModel


@dataclass
class AppState:
    df: object = None
    filename: str = ""
    schema: object = None
    prepared: object = None
    models: dict = field(default_factory=dict)          # key -> TrainedModel
    cost_model: CostModel = field(default_factory=CostModel)
    k_rate: float = 0.01
    thresholds: dict = field(default_factory=dict)      # key -> chosen threshold
    imbalance: str = "class_weight"
    cache: dict = field(default_factory=dict)           # explainers / SHAP results


def S() -> AppState:
    if "app" not in st.session_state:
        st.session_state["app"] = AppState()
    return st.session_state["app"]


def require(cond, msg: str):
    if not cond:
        st.info(msg)
        st.stop()


def reset_downstream(level: str):
    """Invalidate everything computed from a changed upstream step."""
    s = S()
    if level in ("data", "prep"):
        s.models, s.thresholds, s.cache = {}, {}, {}
    if level == "data":
        s.prepared = None
