"""Business-cost threshold: cost = amounts of missed frauds + review cost per alert."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CostModel:
    cost_review: float = 5.0        # analyst cost per alert
    use_amounts: bool = True        # missed fraud costs its transaction amount
    fixed_fn_cost: float = 500.0    # used when no amount column / use_amounts False


def fraud_loss(y, amounts, cm: CostModel) -> np.ndarray:
    """Per-row loss if the row is a missed fraud (0 for legit rows)."""
    y = np.asarray(y)
    if cm.use_amounts and amounts is not None:
        loss = np.clip(np.asarray(amounts, dtype=float), 0, None)
    else:
        loss = np.full(len(y), cm.fixed_fn_cost)
    return loss * (y == 1)


def cost_at(s, thr, loss, cm: CostModel) -> dict:
    alerts = np.asarray(s) >= thr
    missed = float(loss[~alerts].sum())
    review = float(alerts.sum() * cm.cost_review)
    return {"missed_cost": missed, "review_cost": review, "total_cost": missed + review,
            "baseline_no_alert": float(loss.sum())}


def cost_curve(s, loss, cm: CostModel) -> pd.DataFrame:
    """Total cost for every possible number of alerts k (sorted by decreasing score)."""
    s = np.asarray(s)
    order = np.argsort(-s, kind="stable")
    cum = np.concatenate([[0.0], np.cumsum(loss[order])])
    k = np.arange(len(s) + 1)
    thr = np.concatenate([[np.inf], s[order]])
    return pd.DataFrame({"k": k, "threshold": thr, "total_cost": cm.cost_review * k + loss.sum() - cum})


def optimal_threshold(s, loss, cm: CostModel) -> dict:
    cc = cost_curve(s, loss, cm)
    i = int(cc["total_cost"].idxmin())
    thr = float(cc.loc[i, "threshold"])
    if np.isinf(thr):  # best policy = never alert
        thr = float(np.max(s)) + 1e-9
    return {"threshold": thr, "k": int(cc.loc[i, "k"]), **cost_at(s, thr, loss, cm)}
