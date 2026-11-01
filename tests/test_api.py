from fastapi.testclient import TestClient

from fraud_xai.api.main import create_app
from fraud_xai.data.schema import Schema
from fraud_xai.data.synthetic import make_demo_dataset
from fraud_xai.evaluation.threshold import CostModel
from fraud_xai.services.persistence import build_bundle, save_bundle
from fraud_xai.services.pipeline import PrepConfig, prepare, train_model


def test_api_roundtrip(tmp_path):
    df = make_demo_dataset(4000, 0.05)
    prep = prepare(df, Schema("is_fraud", 1, "timestamp", "amount", "card_id"), PrepConfig(use_behavioural=False))
    m = train_model(prep, "xgboost")
    path = save_bundle(build_bundle(prep, m, 0.5, CostModel(), {}), tmp_path / "b.joblib")
    rec = df.drop(columns="is_fraud").iloc[0].to_dict()
    rec["timestamp"] = str(rec["timestamp"])
    with TestClient(create_app(path)) as c:
        assert c.get("/health").json()["model_loaded"] is True
        r = c.post("/predict", json={"records": [rec]})
        assert r.status_code == 200 and 0 <= r.json()[0]["score"] <= 1
        e = c.post("/explain", json={"record": rec, "top_n": 3})
        assert e.status_code == 200 and len(e.json()["reason_codes"]) <= 3
    with TestClient(create_app(tmp_path / "missing.joblib")) as c:
        assert c.post("/predict", json={"records": [rec]}).status_code == 503
