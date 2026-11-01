import numpy as np
import streamlit as st
import joblib
import io

from fraud_xai.config import DEFAULT_BUNDLE
from fraud_xai.evaluation.threshold import fraud_loss, optimal_threshold
from fraud_xai.services.persistence import build_bundle, save_bundle
from ui.state import S, require

s = S()
st.title("🚀 Export & API")
require(s.models, "Train at least one model first (page 3).")
p, cm = s.prepared, s.cost_model

key = st.selectbox("Model to deploy", list(s.models), format_func=lambda k: s.models[k].label,
                   index=int(np.argmax([m.metrics["val"]["pr_auc"] for m in s.models.values()])))
m = s.models[key]
default_thr = s.thresholds.get(key)
if default_thr is None:
    default_thr = optimal_threshold(m.scores["val"], fraud_loss(p.y["val"], p.amounts["val"], cm), cm)["threshold"]
thr = st.number_input("Alert threshold", 0.0, 1.0, float(min(default_thr, 1.0)), step=0.0005, format="%.4f",
                      help="Defaults to the value chosen on page 4.")

if st.button("💾 Save bundle", type="primary"):
    b = build_bundle(p, m, thr, cm, {k: v for k, v in m.metrics.items()})
    path = save_bundle(b, DEFAULT_BUNDLE)
    st.success(f"Saved to `{path}`")
    buf = io.BytesIO(); joblib.dump(b, buf)
    st.download_button("Download bundle", buf.getvalue(), "fraud_bundle.joblib")
    if b["client_supplied_columns"]:
        st.warning("Behavioural features need transaction history: the API caller must send "
                   f"these columns already computed: `{b['client_supplied_columns']}`.")

st.markdown("""
### Run the API
```bash
uvicorn fraud_xai.api.main:app --reload          # docs on http://127.0.0.1:8000/docs
```
| Endpoint | Purpose |
|---|---|
| `GET /health` | service + model status |
| `GET /model` | metadata, threshold, expected columns |
| `POST /predict` | `{"records": [{...}]}` → score, alert, risk level |
| `POST /explain` | `{"record": {...}, "top_n": 4}` → reason codes + SHAP values |
""")
if DEFAULT_BUNDLE.exists():
    from fraud_xai.services.persistence import load_bundle
    b = load_bundle(DEFAULT_BUNDLE)
    st.caption(f"Current bundle: {b['model_label']} · threshold {b['threshold']:.4f} · {b['created_at']}")
    st.write("Expected raw columns:", b["raw_columns"])
