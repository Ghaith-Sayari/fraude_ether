"""Built-in demo dataset so the app runs without any download."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def make_demo_dataset(n: int = 30_000, fraud_rate: float = 0.02, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X, y = make_classification(
        n_samples=n, n_features=12, n_informative=7, n_redundant=2,
        weights=[1 - fraud_rate], class_sep=1.3, flip_y=0.002, random_state=seed,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["is_fraud"] = y
    day = rng.integers(0, 90, n)
    hour = np.where(y == 1, rng.choice([1, 2, 3, 23, 12], n), rng.integers(7, 22, n))
    sec = rng.integers(0, 3600, n)
    df["timestamp"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(day * 86400 + hour * 3600 + sec, unit="s")
    df["amount"] = np.round(np.where(y == 1, rng.lognormal(5.2, 1.0, n), rng.lognormal(3.6, 1.0, n)), 2)
    df["card_id"] = rng.integers(0, 1500, n)
    df["channel"] = rng.choice(["web", "pos", "atm", "mobile"], n, p=[.4, .3, .1, .2])
    df["category"] = rng.choice(["grocery", "travel", "electronics", "fuel", "leisure"], n)
    return df.sort_values("timestamp").reset_index(drop=True)
