import pandas as pd
import streamlit as st

from fraud_xai.features.preprocessing import SCALERS
from fraud_xai.models.imbalance import STRATEGIES
from fraud_xai.models.registry import REGISTRY, specs_by_family
from fraud_xai.services.pipeline import PrepConfig, compare_imbalance, prepare
from fraud_xai.viz import plots
from ui.state import S, require, reset_downstream

s = S()
st.title("🧪 Preparation & features")
require(s.df is not None and s.schema is not None, "Load a dataset first (page 1).")
sch = s.schema

st.subheader("Features")
c1, c2 = st.columns(2)
use_temp = c1.checkbox("Temporal features (cyclic hour, day of week, weekend)", True,
                       disabled=sch.time_col is None)
use_beh = c2.checkbox("Behavioural features (tx count 1h/24h, amount vs 30-day mean, time since previous)",
                      True, disabled=not (sch.time_col and sch.entity_col and sch.amount_col),
                      help="Needs a datetime column, an entity id and an amount. Uses only PAST rows of each entity.")
st.subheader("Preprocessing (fitted on train only)")
c3, c4, c5 = st.columns(3)
scaler = c3.selectbox("Scaling", list(SCALERS), format_func=SCALERS.get)
corr = c4.slider("Drop features correlated above", 0.5, 1.0, 0.95, 0.01,
                 help="Replaces the hand-written drop list of the notebook.")
max_oh = c5.slider("One-hot if ≤ n categories", 2, 50, 10)

if st.button("Run preparation", type="primary"):
    try:
        with st.spinner("Building features, splitting chronologically, fitting preprocessor..."):
            s.prepared = prepare(s.df, sch, PrepConfig(use_temp, use_beh, scaler, corr, max_oh))
        reset_downstream("prep")
    except Exception as e:
        st.error(str(e))

p = s.prepared
require(p is not None, "Click **Run preparation**.")
sp = p.split
if sp.warning:
    st.warning(sp.warning)
else:
    st.success("Chronological split: 60% train → 20% validation → 20% test (no shuffling).")

rows = []
for name, X, y in (("train", sp.X_train, sp.y_train), ("validation", sp.X_val, sp.y_val), ("test", sp.X_test, sp.y_test)):
    rows.append({"split": name, "rows": len(y), "frauds": int(y.sum()), "fraud rate": y.mean()})
st.dataframe(pd.DataFrame(rows).style.format({"fraud rate": "{:.3%}", "rows": "{:,}"}), width="stretch", hide_index=True)

r = p.preprocessor.report_
st.caption(f"{r['input_columns']} raw columns → **{r['output_features']} model features**")
with st.expander("What the preprocessor removed"):
    st.write("Constant:", r["dropped_constant"] or "—")
    st.write("Highly correlated:", r["dropped_correlated"] or "—")
    st.write("High-cardinality text:", r["dropped_high_cardinality"] or "—")

# ---------------- imbalance -----------------
st.subheader("Imbalance strategy (justified on validation)")
sup = [x.key for x in specs_by_family("supervised")]
c1, c2 = st.columns([1, 2])
mkey = c1.selectbox("Reference model", sup, format_func=lambda k: REGISTRY[k].label, index=sup.index("logreg"))
strats = c2.multiselect("Strategies to compare", list(STRATEGIES), default=list(STRATEGIES), format_func=STRATEGIES.get)
if st.button("Compare strategies") and strats:
    with st.spinner("Training one model per strategy (resampling applied to train only)..."):
        s.cache["imb"] = compare_imbalance(p, mkey, {}, strats)
if "imb" in s.cache:
    t = s.cache["imb"]
    st.dataframe(t.style.format({"val_pr_auc": "{:.4f}", "val_roc_auc": "{:.4f}"}), width="stretch", hide_index=True)
    st.plotly_chart(plots.bar_fig(t, "strategy", "val_pr_auc", "Validation PR-AUC by strategy"), width="stretch")
    s.imbalance = t.loc[0, "strategy"]
    st.info(f"Best on validation: **{STRATEGIES[s.imbalance]}** (pre-selected on the next page).")
