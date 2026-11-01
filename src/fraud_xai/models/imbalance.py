"""Imbalance strategies - applied to the TRAINING set only."""
from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

STRATEGIES = {
    "none": "None",
    "class_weight": "Class weights (cost-sensitive)",
    "smote": "SMOTE oversampling",
    "undersample": "Random undersampling",
}


def make_sampler(strategy: str, y, seed: int = 42):
    """Return an imblearn sampler (or None). Safe k_neighbors for tiny minorities."""
    if strategy == "smote":
        n_pos = int(np.sum(y == 1))
        return SMOTE(k_neighbors=max(1, min(5, n_pos - 1)), random_state=seed)
    if strategy == "undersample":
        return RandomUnderSampler(random_state=seed)
    return None


def pos_weight(y) -> float:
    y = np.asarray(y)
    return float((y == 0).sum() / max((y == 1).sum(), 1))
