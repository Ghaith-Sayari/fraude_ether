import numpy as np
import pandas as pd
import streamlit as st

from fraud_xai.models.imbalance import STRATEGIES
from fraud_xai.models.registry import REGISTRY, specs_by_family
from fraud_xai.models.unsupervised import k_distances
from fraud_xai.services.pipeline import train_model, tune_model
from fraud_xai.viz import plots
from ui.components import param_form
from ui.state import S, require

s = S()
st.title("🤖 Models & fine-tuning")
require(s.prepared is not None, "Run the preparation step first (page 2).")
p = s.prepared

c1, c2 = st.columns(2)
sup = c1.multiselect("Supervised models", [x.key for x in specs_by_family("supervised")],
                     default=["logreg", "xgboost"], format_func=lambda k: REGISTRY[k].label)
uns = c2.multiselect("Unsupervised (anomaly) models", [x.key for x in specs_by_family("unsupervised")],
                     default=["iforest", "dbscan"], format_func=lambda k: REGISTRY[k].label)
chosen = sup + uns
require(chosen, "Select at least one model.")

g1, g2, g3 = st.columns(3)
strategy = g1.selectbox("Imbalance strategy (supervised, train only)", list(STRATEGIES),
                        index=list(STRATEGIES).index(s.imbalance), format_func=STRATEGIES.get,
                        help="Compare strategies on page 2 to choose.")
legit_only = g2.checkbox("Unsupervised: fit on legitimate transactions only", False,
                         help="Novelty-detection setup. Uses labels only to filter training rows.")
cv = g3.slider("CV folds (supervised tuning)", 2, 5, 3,
               help="TimeSeriesSplit if the split is chronological, else stratified K-fold.")

settings = {}
for key in chosen:
    spec = REGISTRY[key]
    with st.expander(f"{spec.label}  ·  {spec.family}", expanded=False):
        st.caption(spec.description)
        auto = st.checkbox("Auto-tune with random search", False, key=f"auto_{key}",
                           help="Supervised: CV PR-AUC on train. Unsupervised: PR-AUC on validation.")
        n_iter = st.slider("Search iterations", 5, 60, 15, key=f"iter_{key}") if auto else 0
        manual = param_form(spec, "m")
        if auto:
            st.caption("Manual values are ignored when auto-tune is on.")
        if key == "dbscan":
            k = manual["min_samples"]
            kd = k_distances(p.X_train, k)
            st.plotly_chart(plots.kdist_fig(kd, float(np.quantile(kd, manual["eps_quantile"]))), width="stretch")
        settings[key] = (manual, auto, n_iter)

if st.button("🚀 Train selected models", type="primary"):
    bar, status = st.progress(0.0), st.empty()
    for i, key in enumerate(chosen):
        manual, auto, n_iter = settings[key]
        spec = REGISTRY[key]
        try:
            tr = None
            if auto:
                status.write(f"Tuning **{spec.label}** ({n_iter} candidates)...")
                tr = tune_model(p, key, strategy, n_iter, cv, legit_only)
                manual = tr.best_params
            status.write(f"Training **{spec.label}**...")
            s.models[key] = train_model(p, key, manual, strategy, legit_only, tr)
            s.thresholds.pop(key, None)
            s.cache = {k: v for k, v in s.cache.items() if not (isinstance(k, tuple) and k[0] == key)}
        except Exception as e:
            st.error(f"{spec.label} failed: {e}")
        bar.progress((i + 1) / len(chosen))
    status.empty()

if s.models:
    st.subheader("Results on validation")
    rows = [{"model": m.label, "type": m.family, "imbalance": m.imbalance,
             "val PR-AUC": m.metrics["val"]["pr_auc"], "val ROC-AUC": m.metrics["val"]["roc_auc"],
             "fit time (s)": m.fit_seconds, "tuned": m.tuning is not None, "key": m.key}
            for m in s.models.values()]
    t = pd.DataFrame(rows).sort_values("val PR-AUC", ascending=False)
    st.dataframe(t.drop(columns="key").style.format({"val PR-AUC": "{:.4f}", "val ROC-AUC": "{:.4f}", "fit time (s)": "{:.1f}"}),
                 width="stretch", hide_index=True)
    st.caption("Test metrics are deliberately hidden here — see page 4 once the threshold is chosen on validation.")
    for m in s.models.values():
        with st.expander(f"Final parameters · {m.label}"):
            st.json(m.params)
            if m.tuning is not None:
                st.caption("Search results (best first)")
                st.dataframe(m.tuning.table.head(15), width="stretch", hide_index=True)
            if m.key == "dbscan":
                est = m.detector.estimator
                st.write(f"eps = **{est.eps_:.3f}** · clusters = **{est.n_clusters_}** · "
                         f"noise on train sample = **{est.noise_ratio_:.1%}**")
