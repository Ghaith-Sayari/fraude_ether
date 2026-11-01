import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from fraud_xai.explain.importance import permutation_importance
from fraud_xai.explain.shap_explainer import FraudExplainer, reason_codes
from ui.state import S, require
from fraud_xai.viz import plots

s = S()
st.title("🔍 Explainability")
require(s.models, "Train at least one model first (page 3).")
p = s.prepared

key = st.selectbox("Model to explain", list(s.models), format_func=lambda k: s.models[k].label,
                   index=int(np.argmax([m.metrics["val"]["pr_auc"] for m in s.models.values()])))
m = s.models[key]


def explainer() -> FraudExplainer:
    if (key, "exp") not in s.cache:
        with st.spinner("Building SHAP explainer..."):
            s.cache[(key, "exp")] = FraudExplainer(m.detector, key, p.X_train)
    return s.cache[(key, "exp")]


tab_g, tab_l = st.tabs(["Global", "One alert"])

# ---------------- global -----------------
with tab_g:
    c1, c2 = st.columns(2)
    method = c1.radio("Method", ["SHAP beeswarm", "Permutation importance"], horizontal=True)
    n = c2.slider("Test rows used", 50, 500, 150, 50,
                  help="Model-agnostic SHAP (Isolation Forest, DBSCAN...) is slower: keep this small.")
    if st.button("Compute global explanation"):
        Xs = p.X_test.sample(min(n, len(p.X_test)), random_state=0)
        with st.spinner("Computing..."):
            if method == "SHAP beeswarm":
                s.cache[(key, "glob")] = ("shap", explainer().explain(Xs))
            else:
                ys = p.y["test"].loc[Xs.index]
                s.cache[(key, "glob")] = ("perm", permutation_importance(m.detector, Xs, ys))
    if (key, "glob") in s.cache:
        kind, res = s.cache[(key, "glob")]
        if kind == "shap":
            shap.plots.beeswarm(res, max_display=15, show=False)
            fig = plt.gcf(); fig.set_size_inches(8, 5.5); plt.tight_layout()
            st.pyplot(fig); plt.close(fig)
            st.caption(f"Units: {explainer().units}. Red = high feature value, right = pushes toward fraud.")
        else:
            top = res.head(15).iloc[::-1]
            st.plotly_chart(plots.bar_fig(top, "importance", "feature", "PR-AUC drop when feature is shuffled")
                            .update_traces(orientation="h", x=top["importance"], y=top["feature"]), width="stretch")

# ---------------- local -----------------
with tab_l:
    sc = m.scores["test"]
    thr = s.thresholds.get(key)
    if thr is None:
        st.caption("No threshold chosen yet on page 4: showing the highest-scored test transactions.")
    n_top = st.slider("Show top-N test transactions by score", 5, 100, 20)
    top_idx = np.argsort(-sc)[:n_top]
    tbl = pd.DataFrame({"rank": np.arange(1, n_top + 1), "score": sc[top_idx],
                        "true label": np.where(p.y["test"].to_numpy()[top_idx] == 1, "FRAUD", "legit"),
                        "alert": (sc[top_idx] >= thr) if thr is not None else None})
    st.dataframe(tbl.style.format({"score": "{:.4f}"}), width="stretch", hide_index=True, height=220)
    rank = st.number_input("Explain alert #", 1, n_top, 1)
    pos = int(top_idx[rank - 1])
    row = p.X_test.iloc[[pos]]
    with st.spinner("Explaining..."):
        e = explainer().explain(row)
    vals = e.values[0]
    left, right = st.columns([3, 2])
    with left:
        shap.plots.waterfall(e[0], max_display=12, show=False)
        fig = plt.gcf(); plt.tight_layout()
        st.pyplot(fig); plt.close(fig)
    with right:
        st.metric("Suspicion score", f"{sc[pos]:.4f}")
        st.markdown("**Reason codes**")
        shown = {**row.iloc[0].to_dict(), **p.raw["test"].iloc[pos].to_dict()}
        n_codes = st.slider("Number of codes", 3, 5, 4)
        for i, rc in enumerate(reason_codes(vals, explainer().names, shown, n_codes), 1):
            st.write(f"**R{i}.** {rc['text']}")
        with st.expander("Raw transaction"):
            st.dataframe(p.raw["test"].iloc[[pos]].T, width="stretch")
