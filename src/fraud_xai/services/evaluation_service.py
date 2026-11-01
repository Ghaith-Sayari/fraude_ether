"""Comparison table across all trained models (validation-chosen threshold, test-confirmed)."""
from __future__ import annotations

import pandas as pd

from fraud_xai.evaluation.metrics import at_k, at_threshold, ranking_metrics
from fraud_xai.evaluation.threshold import CostModel, cost_at, fraud_loss, optimal_threshold


def k_for(n: int, k_rate: float) -> int:
    return max(1, int(round(n * k_rate)))


def comparison_table(models: dict, prepared, cm: CostModel, k_rate: float) -> pd.DataFrame:
    rows = []
    yv, yt = prepared.y["val"], prepared.y["test"]
    lv = fraud_loss(yv, prepared.amounts["val"], cm)
    lt = fraud_loss(yt, prepared.amounts["test"], cm)
    for m in models.values():
        sv, st = m.scores["val"], m.scores["test"]
        best = optimal_threshold(sv, lv, cm)            # chosen on validation
        thr = best["threshold"]
        tv, tt = at_threshold(yv, sv, thr), at_threshold(yt, st, thr)
        kt = at_k(yt, st, k_for(len(yt), k_rate))
        rt = ranking_metrics(yt, st)
        rows.append({
            "model": m.label, "type": m.family, "imbalance": m.imbalance,
            "val PR-AUC": m.metrics["val"]["pr_auc"], "test PR-AUC": rt["pr_auc"],
            "test ROC-AUC": rt["roc_auc"],
            f"test P@k": kt["precision_at_k"], f"test R@k": kt["recall_at_k"],
            "threshold (val)": thr, "test F2": tt["f2"], "test MCC": tt["mcc"],
            "test alerts": tt["alerts"],
            "val cost": cost_at(sv, thr, lv, cm)["total_cost"],
            "test cost": cost_at(st, thr, lt, cm)["total_cost"],
            "test cost, no model": float(lt.sum()), "key": m.key,
        })
    return pd.DataFrame(rows).sort_values("test PR-AUC", ascending=False).reset_index(drop=True)
