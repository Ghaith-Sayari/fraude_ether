"""Unified detector interface: every model exposes score(X) -> higher = more suspicious.
Lets supervised and unsupervised methods share evaluation, thresholds and SHAP."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit


class ProbaDetector:
    """Supervised classifier -> fraud probability."""
    family = "supervised"

    def __init__(self, estimator):
        self.estimator = estimator

    def score(self, X) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1]


class AnomalyDetector:
    """Unsupervised estimator -> anomaly score squashed to (0, 1) with a robust
    z-score fitted on training scores, so a single threshold slider works for all models."""
    family = "unsupervised"

    def __init__(self, estimator, fit_on_legit_only: bool = False,
                 max_fit_samples: int = 20_000, seed: int = 42):
        self.estimator = estimator
        self.fit_on_legit_only = fit_on_legit_only
        self.max_fit_samples = max_fit_samples
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "AnomalyDetector":
        if self.fit_on_legit_only and y is not None:
            X = X[np.asarray(y) == 0]
        if len(X) > self.max_fit_samples:
            X = X.sample(self.max_fit_samples, random_state=self.seed)
        self.estimator.fit(X)
        raw = self.raw_score(X)
        q1, med, q3 = np.percentile(raw, [25, 50, 75])
        self.center_ = med
        self.scale_ = (q3 - q1) / 1.349 or raw.std() or 1.0
        return self

    def raw_score(self, X) -> np.ndarray:
        e = self.estimator
        if hasattr(e, "score_samples"):
            return -np.asarray(e.score_samples(X))
        return -np.asarray(e.decision_function(X))

    def score(self, X) -> np.ndarray:
        return expit(np.clip((self.raw_score(X) - self.center_) / self.scale_, -30, 30))
