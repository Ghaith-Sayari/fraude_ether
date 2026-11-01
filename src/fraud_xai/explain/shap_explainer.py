"""SHAP explanations + analyst-friendly 'reason codes'.
tree -> TreeExplainer, linear -> LinearExplainer, others (Isolation Forest, DBSCAN,
LOF, OCSVM, HistGB) -> model-agnostic permutation explainer on detector.score."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from fraud_xai.models.registry import REGISTRY

UNITS = {"tree": "model output (probability / log-odds)", "linear": "log-odds",
         "permutation": "suspicion score (0-1)"}


class FraudExplainer:
    def __init__(self, detector, model_key: str, background: pd.DataFrame,
                 max_background: int = 100, seed: int = 0):
        self.detector = detector
        self.kind = REGISTRY[model_key].explainer
        self.names = list(background.columns)
        bg = background.sample(min(max_background, len(background)), random_state=seed)
        self.background = bg
        est = getattr(detector, "estimator", None)
        if self.kind == "tree":
            self._exp = shap.TreeExplainer(est)
        elif self.kind == "linear":
            self._exp = shap.LinearExplainer(est, bg)
        else:
            f = lambda a: detector.score(pd.DataFrame(a, columns=self.names))
            self._exp = shap.Explainer(f, bg.values, algorithm="permutation")

    @property
    def units(self) -> str:
        return UNITS[self.kind]

    def explain(self, X: pd.DataFrame) -> shap.Explanation:
        X = X[self.names]
        if self.kind == "permutation":
            e = self._exp(X.values, max_evals=2 * len(self.names) + 1)
        else:
            e = self._exp(X)
        vals, base = np.asarray(e.values), np.asarray(e.base_values)
        if vals.ndim == 3:                       # (n, d, classes) -> positive class
            vals, base = vals[:, :, 1], (base[:, 1] if base.ndim == 2 else base)
        if base.ndim == 0:
            base = np.full(len(X), float(base))
        return shap.Explanation(values=vals, base_values=base, data=X.values, feature_names=self.names)


def reason_codes(values: np.ndarray, names: list[str], shown: dict | None = None,
                 top_n: int = 4) -> list[dict]:
    """The 3-5 features pushing this alert UP the most."""
    top_n = int(np.clip(top_n, 1, 8))
    order = np.argsort(-values)
    pick = [i for i in order if values[i] > 0][:top_n] or list(order[:top_n])
    out = []
    for i in pick:
        v = shown.get(names[i]) if shown else None
        out.append({"feature": names[i], "contribution": float(values[i]),
                    "value": None if v is None else (float(v) if isinstance(v, (int, float, np.number)) else str(v)),
                    "text": f"{names[i]}" + (f" = {v:.4g}" if isinstance(v, (int, float, np.number)) else "")
                            + f" (+{values[i]:.3f})"})
    return out
