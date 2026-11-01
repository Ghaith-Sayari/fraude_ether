"""Metrics suited to rare events: PR-AUC, ROC-AUC, precision/recall@k, F2, MCC."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, fbeta_score, matthews_corrcoef, roc_auc_score


def ranking_metrics(y, s) -> dict:
    y = np.asarray(y)
    return {"pr_auc": float(average_precision_score(y, s)),
            "roc_auc": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")}


def at_k(y, s, k: int) -> dict:
    """Precision/recall when analysts review the k highest-scored transactions."""
    y, s = np.asarray(y), np.asarray(s)
    k = int(np.clip(k, 1, len(y)))
    top = np.argsort(-s, kind="stable")[:k]
    tp = int(y[top].sum())
    return {"k": k, "precision_at_k": tp / k, "recall_at_k": tp / max(int(y.sum()), 1)}


def at_threshold(y, s, thr: float) -> dict:
    y, s = np.asarray(y), np.asarray(s)
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return {"threshold": float(thr), "alerts": tp + fp, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
            "mcc": float(matthews_corrcoef(y, pred)) if pred.any() and not pred.all() else 0.0}
