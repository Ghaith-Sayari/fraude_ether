import numpy as np
import pytest

from fraud_xai.data.schema import Schema
from fraud_xai.data.synthetic import make_demo_dataset
from fraud_xai.evaluation.metrics import at_k, at_threshold
from fraud_xai.evaluation.threshold import CostModel, cost_at, fraud_loss, optimal_threshold
from fraud_xai.explain.shap_explainer import FraudExplainer, reason_codes
from fraud_xai.models.registry import REGISTRY
from fraud_xai.services.evaluation_service import comparison_table
from fraud_xai.services.inference import Scorer
from fraud_xai.services.persistence import build_bundle
from fraud_xai.services.pipeline import PrepConfig, compare_imbalance, prepare, train_model, tune_model


@pytest.fixture(scope="module")
def prepared():
    df = make_demo_dataset(6000, 0.05)
    schema = Schema("is_fraud", 1, "timestamp", "amount", "card_id")
    return prepare(df, schema, PrepConfig())


def test_chronological_split_no_overlap(prepared):
    s = prepared.split
    assert s.chronological and len(s.X_train) == 3600
    assert prepared.X_train.shape[1] == prepared.X_test.shape[1]
    assert {"tx_count_1h", "hour_sin"} <= set(s.X_train.columns)


def test_cost_threshold_beats_no_model(prepared):
    cm = CostModel(cost_review=5)
    m = train_model(prepared, "logreg", strategy="class_weight")
    loss = fraud_loss(prepared.y["val"], prepared.amounts["val"], cm)
    best = optimal_threshold(m.scores["val"], loss, cm)
    assert best["total_cost"] <= best["baseline_no_alert"]
    again = cost_at(m.scores["val"], best["threshold"], loss, cm)
    assert again["total_cost"] == pytest.approx(best["total_cost"])


@pytest.mark.parametrize("key", list(REGISTRY))
def test_every_model_trains_and_scores(prepared, key):
    m = train_model(prepared, key)
    assert m.metrics["val"]["pr_auc"] > 0
    assert np.all((m.scores["test"] >= 0) & (m.scores["test"] <= 1))


def test_tuning_and_imbalance(prepared):
    t = tune_model(prepared, "rf", "smote", n_iter=2, cv=2)
    assert "n_estimators" in t.best_params and isinstance(t.best_params["n_estimators"], int)
    tu = tune_model(prepared, "iforest", "none", n_iter=2, cv=2)
    assert isinstance(tu.best_params["n_estimators"], int)
    cmp = compare_imbalance(prepared, "logreg", {}, ["none", "class_weight", "smote", "undersample"])
    assert len(cmp) == 4


def test_metrics(prepared):
    y = prepared.y["val"].values
    perfect = y.astype(float)
    assert at_k(y, perfect, int(y.sum()))["precision_at_k"] == 1.0
    assert at_threshold(y, perfect, 0.5)["f2"] == 1.0


@pytest.mark.parametrize("key", ["xgboost", "iforest", "logreg"])
def test_shap_and_reason_codes(prepared, key):
    m = train_model(prepared, key)
    ex = FraudExplainer(m.detector, key, prepared.X_train, max_background=30)
    e = ex.explain(prepared.X_test.iloc[:3])
    assert e.values.shape == (3, prepared.X_test.shape[1])
    assert 1 <= len(reason_codes(e.values[0], ex.names, top_n=4)) <= 4


def test_bundle_scorer_and_comparison(prepared):
    cm = CostModel()
    models = {k: train_model(prepared, k) for k in ("xgboost", "dbscan")}
    tab = comparison_table(models, prepared, cm, 0.05)
    assert len(tab) == 2
    b = build_bundle(prepared, models["xgboost"], 0.5, cm, {})
    sc = Scorer(b)
    rec = {"f0": 0.1, "amount": 100.0, "timestamp": "2025-03-01 02:00:00",
           "channel": "web", "category": "fuel", "card_id": 5,
           "tx_count_1h": 1, "tx_count_24h": 2, "amount_vs_30d_mean": 1.0, "secs_since_prev_tx": 100.0}
    out = sc.explain(rec)
    assert 0 <= out["score"] <= 1 and out["reason_codes"]
