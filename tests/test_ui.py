"""Smoke-test every Streamlit page headlessly with scripted state."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from fraud_xai.data.schema import Schema
from fraud_xai.data.synthetic import make_demo_dataset
from fraud_xai.services.pipeline import PrepConfig, prepare, train_model
from ui.state import AppState


@pytest.fixture(scope="module")
def state():
    s = AppState()
    s.df, s.filename = make_demo_dataset(5000, 0.05), "demo"
    s.schema = Schema("is_fraud", 1, "timestamp", "amount", "card_id")
    s.prepared = prepare(s.df, s.schema, PrepConfig())
    for k in ("xgboost", "iforest", "dbscan"):
        s.models[k] = train_model(s.prepared, k)
    return s


@pytest.mark.parametrize("page", ["data", "preparation", "modeling", "evaluation", "explain", "export"])
def test_page_renders(page, state):
    at = AppTest.from_file(str(Path(__file__).parents[1] / "views" / f"{page}.py"), default_timeout=120)
    at.session_state["app"] = state
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def _page(name):
    return str(Path(__file__).parents[1] / "views" / f"{name}.py")


def test_prepare_compare_train_flow():
    import numpy as np
    import pandas as pd
    s = AppState()
    # Ethereum-like: no time column, address string column, NaNs -> random split path
    df = make_demo_dataset(4000, 0.1).drop(columns=["timestamp", "card_id"])
    df["Address"] = [f"0x{i:08x}" for i in range(len(df))]
    df.loc[df.sample(200, random_state=1).index, "f0"] = np.nan
    s.df, s.filename = df, "eth_like"
    s.schema = Schema("is_fraud", 1, None, "amount", None, ["Address"])

    at = AppTest.from_file(_page("preparation"), default_timeout=180)
    at.session_state["app"] = s
    at.run()
    at.button[0].click().run()                       # Run preparation
    assert not at.exception, [e.value for e in at.exception]
    assert s.prepared is not None and not s.prepared.split.chronological
    assert any("RANDOM" in w.value for w in at.warning)
    at.button[1].click().run()                       # Compare strategies
    assert not at.exception, [e.value for e in at.exception]
    assert "imb" in s.cache and len(s.cache["imb"]) == 4

    at = AppTest.from_file(_page("modeling"), default_timeout=300)
    at.session_state["app"] = s
    at.run()
    at.checkbox(key="auto_xgboost").check().run()    # auto-tune xgboost
    at.slider(key="iter_xgboost").set_value(5).run()
    at.button[0].click().run()                       # Train
    assert not at.exception, [e.value for e in at.exception]
    assert {"logreg", "xgboost", "iforest", "dbscan"} <= set(s.models)
    assert s.models["xgboost"].tuning is not None

    at = AppTest.from_file(_page("explain"), default_timeout=300)
    at.session_state["app"] = s
    at.run()
    [b for b in at.button if "global" in b.label][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(k[1] == "glob" for k in s.cache if isinstance(k, tuple))
