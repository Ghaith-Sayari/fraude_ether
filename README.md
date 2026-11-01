# 🛡️ Explainable Fraud Detection – Streamlit app + FastAPI

Turns the *Ethereum fraud* notebook into a layered application: import **any** fraud dataset, choose and
fine-tune supervised **and** unsupervised models (Isolation Forest, DBSCAN, LOF, One-Class SVM), pick the alert threshold
from a business cost, and explain every alert with SHAP.

## Quick start
```bash
pip install -r requirements.txt && pip install -e .
streamlit run app.py                                   # UI
uvicorn fraud_xai.api.main:app --reload               # API (after "Export" page)
pytest                                                 # 24 tests (core, API, every Streamlit page)
```

## Architecture
```
app.py                     Streamlit entrypoint (st.navigation, pipeline status sidebar)
views/                     UI only – one file per page, no ML logic
  data.py  preparation.py  modeling.py  evaluation.py  explain.py  export.py
ui/                        state.py (typed session state) · components.py (param forms)
src/fraud_xai/
  config.py                paths, seeds, 60/20/20 ratios
  data/                    loader · schema (target/id heuristics) · splitter (chrono) · synthetic demo
  features/                engineering (temporal + behavioural) · preprocessing (train-only fit)
  models/                  registry (ONE spec per model → UI form + search space + builder)
                           base (unified score()) · unsupervised (DBSCAN wrapper) · imbalance · tuning
  evaluation/              metrics (PR-AUC, P/R@k, F2, MCC) · threshold (business cost)
  explain/                 shap_explainer (+ reason codes) · importance (permutation)
  services/                pipeline (use-cases) · evaluation_service · persistence · inference
  viz/plots.py             Plotly figure factories
  api/main.py              FastAPI: /health /model /predict /explain
tests/                     core · api · ui (AppTest on every page + full interaction flow)
artifacts/                 exported model bundles (joblib)
```
Dependency rule: `views → services → (data, features, models, evaluation, explain)`; the lower layers never import Streamlit.

**Add a model** = append one `ModelSpec` in `models/registry.py`; it automatically gets a parameter form,
random-search tuning, comparison-table row, SHAP support and API export.

## What changed vs. the notebook
| Notebook | App |
|---|---|
| Median imputation on the whole dataset | Imputation, scaling, correlation filter **fitted on train only** |
| Hand-written list of columns to drop | Automatic constant + correlation (threshold slider) filtering, id/hash detection |
| Random 80/20 split | Chronological 60/20/20 when a time column exists; stratified random split **with a warning** otherwise |
| SMOTE only | none / class weights / SMOTE / undersampling, compared on **validation PR-AUC**, applied to train only |
| LR, RF, XGBoost, accuracy/recall | + HistGB, Isolation Forest, DBSCAN, LOF, OCSVM; PR-AUC, ROC-AUC, P@k, R@k, F2, MCC |
| Default threshold 0.5 | Threshold minimising **missed-fraud amounts + review cost** on validation, confirmed on test |
| No explanations | SHAP beeswarm / permutation importance, SHAP waterfall, 3-5 reason codes per alert |

## Design notes
* **Unified `score()`**: unsupervised scores are squashed to (0,1) with a robust z-score fitted on training scores, so one threshold slider works for every model.
* **DBSCAN** has no `predict`: it is fitted on a training sample and new rows are scored by distance to the nearest core point. `eps` is data-driven (quantile of the k-distance curve, shown in the UI).
* **Tuning**: supervised = `RandomizedSearchCV` on train (PR-AUC; `TimeSeriesSplit` if chronological; resampler inside an imblearn pipeline so SMOTE never leaks into CV folds). Unsupervised = random search scored by validation PR-AUC (labels used only to *select* hyper-parameters).
* **Behavioural features** only use strictly earlier transactions of the same entity. In the API they must be sent pre-computed (needs history); the `Export` page tells you which.
* Test metrics are hidden on the modeling page and only appear on the evaluation page, after the threshold is set on validation.

## Using the Ethereum dataset from the notebook
Upload `transaction_dataset.csv`, target = `FLAG`, no time column (random split, flagged), amount column optional
(`total ether received` can serve for the cost). `Index`/`Address` are auto-suggested for dropping.
