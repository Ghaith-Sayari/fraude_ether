"""Chronological 60/20/20 split (or stratified random if no time information)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_xai.config import RANDOM_STATE, SPLIT_RATIOS


@dataclass
class Split:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    chronological: bool
    warning: str | None = None


def split_dataset(X: pd.DataFrame, y: pd.Series, time: pd.Series | None = None,
                  ratios=SPLIT_RATIOS, seed: int = RANDOM_STATE) -> Split:
    tr, va, _ = ratios
    if time is not None and time.notna().all():
        order = np.argsort(time.to_numpy(), kind="stable")
        X, y = X.iloc[order], y.iloc[order]
        n = len(X)
        i1, i2 = int(n * tr), int(n * (tr + va))
        parts = (X.iloc[:i1], X.iloc[i1:i2], X.iloc[i2:], y.iloc[:i1], y.iloc[i1:i2], y.iloc[i2:])
        chrono, warn = True, None
    else:
        X_tr, X_rest, y_tr, y_rest = train_test_split(
            X, y, train_size=tr, stratify=y, random_state=seed)
        X_va, X_te, y_va, y_te = train_test_split(
            X_rest, y_rest, train_size=va / (1 - tr), stratify=y_rest, random_state=seed)
        parts = (X_tr, X_va, X_te, y_tr, y_va, y_te)
        chrono = False
        warn = ("No usable time information: stratified RANDOM split used. "
                "Performance may be optimistic compared with real chronological deployment.")
    split = Split(*parts, chronological=chrono, warning=warn)
    for name, yy in (("train", split.y_train), ("validation", split.y_val), ("test", split.y_test)):
        if yy.sum() == 0:
            raise ValueError(f"The {name} split contains no fraud case. "
                             "Choose another time column, or use a random split.")
    return split
