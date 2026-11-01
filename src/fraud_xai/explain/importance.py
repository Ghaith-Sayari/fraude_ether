"""Model-agnostic permutation importance (PR-AUC drop) - works for every detector."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def permutation_importance(detector, X: pd.DataFrame, y, n_repeats: int = 3,
                           max_rows: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if len(X) > max_rows:
        idx = rng.choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], np.asarray(y)[idx]
    y = np.asarray(y)
    base = average_precision_score(y, detector.score(X))
    rows = []
    for c in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[c] = rng.permutation(Xp[c].values)
            drops.append(base - average_precision_score(y, detector.score(Xp)))
        rows.append((c, float(np.mean(drops)), float(np.std(drops))))
    return pd.DataFrame(rows, columns=["feature", "importance", "std"]).sort_values(
        "importance", ascending=False).reset_index(drop=True)
