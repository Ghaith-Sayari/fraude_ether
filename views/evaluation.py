import numpy as np
import pandas as pd
import streamlit as st

from fraud_xai.evaluation.metrics import at_k, at_threshold
from fraud_xai.evaluation.threshold import CostModel, cost_at, cost_curve, fraud_loss, optimal_threshold
from fraud_xai.services.evaluation_service import comparison_table, k_for
from fraud_xai.viz import plots
from ui.components import fmt_money
from ui.state import S, require

s = S()
st.title("📊 Evaluation & alert threshold")
require(s.models, "Train at least one model first (page 3).")
p = s.prepared
yv, yt = p.y["val"].to_numpy(), p.y["test"].to_numpy()

# ---------------- business settings -----------------
with st.sidebar:
    st.subheader("Business settings")
    has_amt = p.schema.amount_col is not None
    review = st.number_input("Cost to review one alert", 0.0, 1e6, float(s.cost_model.cost_review), step=1.0)
    use_amt = st.checkbox("Missed fraud costs its transaction amount", has_amt and s.cost_model.use_amounts,
                          disabled=not has_amt)
    fixed = st.number_input("Fixed cost of a missed fraud", 0.0, 1e9, float(s.cost_model.fixed_fn_cost), step=50.0,
                            disabled=use_amt)
    k_pct = st.slider("Analyst budget k (% of transactions)", 0.1, 20.0, s.k_rate * 100, 0.1,
                      help="How many top-ranked alerts analysts can review per period. "
                           "A natural value is the expected fraud rate.")
    s.cost_model = CostModel(review, use_amt and has_amt, fixed)
    s.k_rate = k_pct / 100
cm = s.cost_model
lv = fraud_loss(yv, p.amounts["val"], cm)
lt = fraud_loss(yt, p.amounts["test"], cm)

# ---------------- comparison -----------------
st.subheader("Comparison of all methods")
st.caption("Threshold = cost-optimal on **validation**; every column prefixed `test` is the independent confirmation.")
tab = comparison_table(s.models, p, cm, s.k_rate)
fmt = {c: "{:.4f}" for c in tab.columns if "AUC" in c or "F2" in c or "MCC" in c or "P@" in c or "R@" in c}
fmt.update({"threshold (val)": "{:.4f}", "val cost": "{:,.0f}", "test cost": "{:,.0f}", "test cost, no model": "{:,.0f}"})
st.dataframe(tab.drop(columns="key").style.format(fmt).background_gradient(subset=["test PR-AUC"], cmap="Greens"),
             width="stretch", hide_index=True)
st.caption(f"P@k / R@k use k = {k_for(len(yt), s.k_rate):,} alerts on the test set.")

# ---------------- PR curves -----------------
st.subheader("Precision–recall curves")
c1, c2 = st.columns([3, 1])
sel = c1.multiselect("Models", list(s.models), default=list(s.models), format_func=lambda k: s.models[k].label)
split = c2.radio("Split", ["val", "test"], horizontal=True)
if sel:
    y_ = p.y[split].to_numpy()
    st.plotly_chart(plots.pr_curves_fig({s.models[k].label: (y_, s.models[k].scores[split]) for k in sel}, y_.mean()),
                    width="stretch")

# ---------------- deep dive -----------------
st.subheader("Interactive threshold & budget")
key = st.selectbox("Model", list(s.models), format_func=lambda k: s.models[k].label,
                   index=int(np.argmax([m.metrics["val"]["pr_auc"] for m in s.models.values()])))
m = s.models[key]
sv, st_ = m.scores["val"], m.scores["test"]
opt = optimal_threshold(sv, lv, cm)
STEP = 0.0005
opt_thr = float(np.clip(np.floor(opt["threshold"] / STEP) * STEP, 0.0, 1.0))
wkey = f"thr_{key}_{review}_{use_amt}_{fixed}"

def _reset(): st.session_state[wkey] = opt_thr
if wkey not in st.session_state:
    st.session_state[wkey] = s.thresholds.get(key, opt_thr)
a, b = st.columns([4, 1])
thr = a.slider("Alert threshold (score ≥ threshold ⇒ alert)", 0.0, 1.0, step=STEP, format="%.4f", key=wkey)
b.button("↺ cost-optimal (validation)", on_click=_reset)
s.thresholds[key] = thr

tv, tt = at_threshold(yv, sv, thr), at_threshold(yt, st_, thr)
cv_, ct_ = cost_at(sv, thr, lv, cm), cost_at(st_, thr, lt, cm)
cc = cost_curve(sv, lv, cm)
cur_k = int((sv >= thr).sum())

for name, t_, c_, base in (("Validation", tv, cv_, lv.sum()), ("Test (confirmation)", tt, ct_, lt.sum())):
    st.markdown(f"**{name}**")
    cols = st.columns(6)
    cols[0].metric("Alerts", f"{t_['alerts']:,}")
    cols[1].metric("Precision", f"{t_['precision']:.1%}")
    cols[2].metric("Recall", f"{t_['recall']:.1%}")
    cols[3].metric("F2 / MCC", f"{t_['f2']:.2f} / {t_['mcc']:.2f}")
    cols[4].metric("Total cost", fmt_money(c_["total_cost"]),
                   delta=f"{c_['total_cost'] - base:,.0f} vs no model", delta_color="inverse")
    cols[5].metric("Missed / review", f"{fmt_money(c_['missed_cost'])} / {fmt_money(c_['review_cost'])}")

g1, g2, g3 = st.columns([2, 1, 1])
g1.plotly_chart(plots.cost_curve_fig(cc, cur_k, opt["k"]), width="stretch")
g2.plotly_chart(plots.confusion_fig(tv["tp"], tv["fp"], tv["fn"], tv["tn"], "Validation"), width="stretch")
g3.plotly_chart(plots.confusion_fig(tt["tp"], tt["fp"], tt["fn"], tt["tn"], "Test"), width="stretch")

st.markdown("**Budget view** (top-k ranked alerts)")
kv, kt = k_for(len(yv), s.k_rate), k_for(len(yt), s.k_rate)
rv, rt = at_k(yv, sv, kv), at_k(yt, st_, kt)
d = pd.DataFrame([
    {"split": "validation", "k": kv, "precision@k": rv["precision_at_k"], "recall@k": rv["recall_at_k"]},
    {"split": "test", "k": kt, "precision@k": rt["precision_at_k"], "recall@k": rt["recall_at_k"]}])
st.dataframe(d.style.format({"precision@k": "{:.1%}", "recall@k": "{:.1%}"}), width="stretch", hide_index=True)
