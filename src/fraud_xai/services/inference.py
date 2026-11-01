"""Score / explain raw JSON records with a saved bundle (used by the FastAPI service)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_xai.explain.shap_explainer import FraudExplainer, reason_codes
from fraud_xai.features.engineering import engineer_features


class Scorer:
    def __init__(self, bundle: dict):
        self.b = bundle
        self._explainer = None

    def _features(self, records: list[dict]):
        raw = pd.DataFrame(records)
        eng = engineer_features(raw, self.b["feature_cfg"], behavioural=False)
        return eng, self.b["preprocessor"].transform(eng)

    def predict(self, records: list[dict]) -> pd.DataFrame:
        _, X = self._features(records)
        s = self.b["detector"].score(X)
        thr = self.b["threshold"]
        level = np.where(s >= thr, "high", np.where(s >= 0.7 * thr, "medium", "low"))
        return pd.DataFrame({"score": s, "is_alert": s >= thr, "risk_level": level})

    def explain(self, record: dict, top_n: int = 4) -> dict:
        if self._explainer is None:
            self._explainer = FraudExplainer(self.b["detector"], self.b["model_key"], self.b["background"])
        eng, X = self._features([record])
        e = self._explainer.explain(X)
        vals = e.values[0]
        shown = eng.iloc[0].to_dict()
        pred = self.predict([record]).iloc[0]
        return {"score": float(pred["score"]), "is_alert": bool(pred["is_alert"]),
                "risk_level": str(pred["risk_level"]), "units": self._explainer.units,
                "reason_codes": reason_codes(vals, self._explainer.names, shown, top_n),
                "shap_values": dict(zip(self._explainer.names, map(float, vals)))}
