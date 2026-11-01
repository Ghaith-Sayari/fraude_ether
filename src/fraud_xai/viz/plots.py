"""Plotly figure factories (pure functions: data in, figure out)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve

RED, GREY, BLUE = "#d94f3d", "#8a94a6", "#2f6fdb"
_LAYOUT = dict(margin=dict(l=10, r=10, t=40, b=10), template="plotly_white")


def _fin(fig, title=None, h=360):
    fig.update_layout(title=title, height=h, **_LAYOUT)
    return fig


def class_balance_fig(y: pd.Series):
    c = y.value_counts().rename({0: "Legit", 1: "Fraud"}).reset_index()
    c.columns = ["class", "count"]
    return _fin(px.pie(c, names="class", values="count", hole=.5, color="class",
                       color_discrete_map={"Legit": GREY, "Fraud": RED}), "Class distribution")


def amount_by_class_fig(amount: pd.Series, y: pd.Series):
    d = pd.DataFrame({"amount": pd.to_numeric(amount, errors="coerce"),
                      "class": y.map({0: "Legit", 1: "Fraud"})}).dropna()
    d = d[d["amount"] > 0]
    return _fin(px.box(d, x="class", y="amount", log_y=True, color="class",
                       color_discrete_map={"Legit": GREY, "Fraud": RED}), "Amount by class (log scale)")


def rate_over_time_fig(t: pd.Series, y: pd.Series, bins: int = 30):
    d = pd.DataFrame({"t": t, "y": y}).dropna()
    if pd.api.types.is_datetime64_any_dtype(d["t"]):
        d["bucket"] = pd.cut(d["t"].astype("int64"), bins, labels=False)
        agg = d.groupby("bucket").agg(t=("t", "min"), rate=("y", "mean"), n=("y", "size"))
    else:
        d["bucket"] = pd.cut(d["t"], bins, labels=False)
        agg = d.groupby("bucket").agg(t=("t", "min"), rate=("y", "mean"), n=("y", "size"))
    return _fin(px.line(agg, x="t", y="rate", markers=True, labels={"rate": "fraud rate"}),
                "Fraud rate over time")


def rate_by_category_fig(x: pd.Series, y: pd.Series, top: int = 15, title: str | None = None):
    d = pd.DataFrame({"x": x.astype(str), "y": y}).groupby("x")["y"].agg(["mean", "size"])
    d = d[d["size"] >= 5].sort_values("mean", ascending=False).head(top).reset_index()
    return _fin(px.bar(d, x="x", y="mean", hover_data=["size"], labels={"mean": "fraud rate", "x": ""},
                       color_discrete_sequence=[RED]), title or "Fraud rate by category")


def pr_curves_fig(curves: dict, prevalence: float):
    """curves: label -> (y_true, scores)"""
    fig = go.Figure()
    for lab, (y, s) in curves.items():
        p, r, _ = precision_recall_curve(y, s)
        fig.add_scatter(x=r, y=p, mode="lines", name=lab)
    fig.add_hline(y=prevalence, line_dash="dot", line_color=GREY, annotation_text="random")
    fig.update_xaxes(title="Recall", range=[0, 1]); fig.update_yaxes(title="Precision", range=[0, 1.02])
    return _fin(fig, "Precision-Recall curves", 420)


def cost_curve_fig(cc: pd.DataFrame, cur_k: int, opt_k: int, max_points: int = 1500):
    step = max(1, len(cc) // max_points)
    d = cc.iloc[::step]
    fig = go.Figure(go.Scatter(x=d["k"], y=d["total_cost"], mode="lines", line=dict(color=BLUE)))
    fig.add_vline(x=opt_k, line_color="green", line_dash="dash", annotation_text="optimum")
    fig.add_vline(x=cur_k, line_color=RED, annotation_text="current")
    fig.update_xaxes(title="Number of alerts k"); fig.update_yaxes(title="Total cost")
    return _fin(fig, "Business cost vs number of alerts (validation)")


def confusion_fig(tp: int, fp: int, fn: int, tn: int, title: str):
    z = [[tn, fp], [fn, tp]]
    fig = go.Figure(go.Heatmap(z=z, x=["Pred legit", "Pred fraud"], y=["Legit", "Fraud"],
                               text=z, texttemplate="%{text}", colorscale="Reds", showscale=False))
    fig.update_yaxes(autorange="reversed")
    return _fin(fig, title, 280)


def kdist_fig(kd: np.ndarray, eps: float | None = None):
    fig = go.Figure(go.Scatter(y=kd, mode="lines", line=dict(color=BLUE)))
    if eps:
        fig.add_hline(y=eps, line_color=RED, annotation_text=f"eps = {eps:.3g}")
    fig.update_xaxes(title="Points (sorted)"); fig.update_yaxes(title="k-distance")
    return _fin(fig, "k-distance curve (DBSCAN eps choice)", 300)


def bar_fig(df: pd.DataFrame, x: str, y: str, title: str):
    return _fin(px.bar(df, x=x, y=y, color_discrete_sequence=[BLUE], text_auto=".3f"), title, 300)
