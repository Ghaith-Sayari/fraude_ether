"""Export a trained model + preprocessing + threshold as ONE bundle, used by the API."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib

from fraud_xai.config import DEFAULT_BUNDLE
from fraud_xai.features.engineering import BEHAVIOURAL_COLS


def build_bundle(prepared, model, threshold: float, cost_model, metrics: dict) -> dict:
    fcfg = prepared.feature_cfg
    behav_used = [c for c in BEHAVIOURAL_COLS if c in prepared.split.X_train.columns]
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_key": model.key, "model_label": model.label, "params": model.params,
        "detector": model.detector, "preprocessor": prepared.preprocessor,
        "feature_cfg": fcfg, "threshold": float(threshold), "cost_model": cost_model,
        "metrics": metrics, "feature_names": list(prepared.X_train.columns),
        "raw_columns": list(prepared.split.X_train.columns),
        "client_supplied_columns": behav_used,   # behavioural features the caller must send
        "background": prepared.X_train.sample(min(200, len(prepared.X_train)), random_state=0),
    }


def save_bundle(bundle: dict, path: str | Path = DEFAULT_BUNDLE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: str | Path = DEFAULT_BUNDLE) -> dict:
    return joblib.load(path)
