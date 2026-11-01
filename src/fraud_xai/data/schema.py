"""Dataset schema: which column is what. Includes target / id heuristics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from fraud_xai.config import AMOUNT_CANDIDATES, TARGET_CANDIDATES, TIME_CANDIDATES


@dataclass
class Schema:
    target: str
    positive_label: Any = 1
    time_col: str | None = None
    amount_col: str | None = None
    entity_col: str | None = None
    drop_cols: list[str] = field(default_factory=list)


def _match(df: pd.DataFrame, candidates) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


def guess_target(df: pd.DataFrame) -> str:
    hit = _match(df, TARGET_CANDIDATES)
    if hit:
        return hit
    for c in df.columns[::-1]:
        s = df[c].dropna()
        if s.nunique() == 2 and s.value_counts(normalize=True).min() < 0.3:
            return c
    return df.columns[-1]


def guess_time(df: pd.DataFrame, target: str) -> str | None:
    hit = _match(df, TIME_CANDIDATES)
    return hit if hit != target else None


def guess_amount(df: pd.DataFrame) -> str | None:
    return _match(df, AMOUNT_CANDIDATES)


def suggest_drop_columns(df: pd.DataFrame, target: str) -> list[str]:
    """Identifier-like / constant columns that should not feed a model."""
    out = []
    n = len(df)
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        numeric = pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)
        if c.lower().startswith("unnamed") or c.lower() in {"index", "id"}:
            out.append(c)
        elif not numeric and s.nunique() > 0.5 * n:  # addresses, hashes, names...
            out.append(c)
        elif s.nunique(dropna=False) <= 1:
            out.append(c)
    return out


def binarize_target(s: pd.Series, positive_label: Any) -> pd.Series:
    return (s.astype(str) == str(positive_label)).astype(int).rename(s.name)


def parse_time(s: pd.Series) -> pd.Series:
    """Datetime if parsable, else numeric (e.g. PaySim 'step', BAF 'month')."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_datetime(s, errors="coerce")


def class_balance(y: pd.Series) -> dict:
    pos = int(y.sum())
    return {"n": len(y), "fraud": pos, "legit": len(y) - pos,
            "fraud_rate": pos / max(len(y), 1)}
