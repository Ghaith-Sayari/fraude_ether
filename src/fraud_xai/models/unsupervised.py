"""DBSCAN is transductive (no predict). This wrapper makes it usable on new data:
fit DBSCAN on a training sample, keep its core points, and score a new row by the
distance to the nearest core point (inside eps => 'normal', far => anomalous)."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def k_distances(X, k: int, max_samples: int = 5000, seed: int = 0) -> np.ndarray:
    """Sorted distance to the k-th neighbour: classic plot to pick DBSCAN's eps."""
    X = np.asarray(X)
    if len(X) > max_samples:
        X = X[np.random.default_rng(seed).choice(len(X), max_samples, replace=False)]
    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    return np.sort(d[:, -1])


class DBSCANDetector(BaseEstimator):
    """eps is data-driven: the `eps_quantile` of the k-distance curve (k = min_samples)."""

    def __init__(self, min_samples: int = 10, eps_quantile: float = 0.9):
        self.min_samples = min_samples
        self.eps_quantile = eps_quantile

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.eps_ = float(np.quantile(k_distances(X, self.min_samples), self.eps_quantile))
        db = DBSCAN(eps=self.eps_, min_samples=self.min_samples, n_jobs=-1).fit(X)
        self.labels_ = db.labels_
        self.n_clusters_ = int(len(set(db.labels_)) - (1 if -1 in db.labels_ else 0))
        self.noise_ratio_ = float((db.labels_ == -1).mean())
        core = X[db.core_sample_indices_] if len(db.core_sample_indices_) else X
        self._nn = NearestNeighbors(n_neighbors=1).fit(core)
        return self

    def score_samples(self, X):
        """Negative distance to the nearest core point (sklearn convention: high = normal)."""
        d, _ = self._nn.kneighbors(np.asarray(X, dtype=float))
        return -d[:, 0]
