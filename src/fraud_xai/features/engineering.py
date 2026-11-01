"""Temporal + behavioural features. Behavioural features only look at the PAST
of each entity (strictly earlier rows), so computing them before the split
does not leak future information."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraud_xai.data.schema import parse_time

HOUR_NS = 3600 * 10**9
BEHAVIOURAL_COLS = ["tx_count_1h", "tx_count_24h", "amount_vs_30d_mean", "secs_since_prev_tx"]


@dataclass
class FeatureConfig:
    time_col: str | None = None
    amount_col: str | None = None
    entity_col: str | None = None
    use_temporal: bool = True
    use_behavioural: bool = True


def _behavioural(t_ns: np.ndarray, amount: np.ndarray, entity: np.ndarray) -> dict[str, np.ndarray]:
    n = len(t_ns)
    c1, c24 = np.zeros(n), np.zeros(n)
    ratio, gap = np.ones(n), np.full(n, np.nan)
    codes = pd.factorize(entity)[0]
    order = np.lexsort((t_ns, codes))
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    for idx in np.split(order, bounds):
        t, a = t_ns[idx], amount[idx]
        pos = np.arange(len(idx))
        c1[idx] = pos - np.searchsorted(t, t - HOUR_NS, "left")
        c24[idx] = pos - np.searchsorted(t, t - 24 * HOUR_NS, "left")
        left30 = np.searchsorted(t, t - 30 * 24 * HOUR_NS, "left")
        cs = np.concatenate([[0.0], np.cumsum(a)])
        n_prev = pos - left30
        mean_prev = np.divide(cs[pos] - cs[left30], n_prev, out=np.zeros(len(idx)), where=n_prev > 0)
        ratio[idx] = np.where(n_prev > 0, a / np.where(mean_prev == 0, 1, mean_prev), 1.0)
        gap[idx[1:]] = np.diff(t) / 1e9
    return {"tx_count_1h": c1, "tx_count_24h": c24,
            "amount_vs_30d_mean": ratio, "secs_since_prev_tx": gap}


def engineer_features(X: pd.DataFrame, cfg: FeatureConfig, behavioural: bool = True) -> pd.DataFrame:
    """Return model-ready raw features. time/entity ids are dropped afterwards.
    At inference time call with behavioural=False (client sends those columns)."""
    out = X.copy()
    drop = []
    t = parse_time(out[cfg.time_col]) if cfg.time_col and cfg.time_col in out else None
    is_dt = t is not None and pd.api.types.is_datetime64_any_dtype(t)

    if is_dt and cfg.use_temporal:
        out["hour_sin"] = np.sin(2 * np.pi * t.dt.hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * t.dt.hour / 24)
        out["day_of_week"] = t.dt.dayofweek
        out["is_weekend"] = (t.dt.dayofweek >= 5).astype(int)

    can_behave = (is_dt and cfg.use_behavioural and behavioural and cfg.entity_col
                  and cfg.entity_col in out and cfg.amount_col and cfg.amount_col in out)
    if can_behave:
        feats = _behavioural(t.astype("int64").to_numpy(),
                             pd.to_numeric(out[cfg.amount_col], errors="coerce").fillna(0).to_numpy(),
                             out[cfg.entity_col].to_numpy())
        for k, v in feats.items():
            out[k] = v

    for c in (cfg.time_col, cfg.entity_col):
        if c and c in out:
            drop.append(c)
    return out.drop(columns=drop)
