"""Hyper-parameter search.
- supervised: RandomizedSearchCV on TRAIN only (scoring = PR-AUC / average precision),
  stratified folds, or TimeSeriesSplit when the split is chronological. The resampler
  lives inside an imblearn Pipeline so SMOTE never touches validation folds.
- unsupervised: random search scored by PR-AUC on VALIDATION (labels used only to
  select hyper-parameters, never to fit)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, StratifiedKFold, TimeSeriesSplit

from fraud_xai.models.base import AnomalyDetector
from fraud_xai.models.imbalance import make_sampler, pos_weight
from fraud_xai.models.registry import ModelSpec


@dataclass
class TuneResult:
    best_params: dict
    best_score: float
    table: pd.DataFrame


def tune_supervised(spec: ModelSpec, X, y, strategy: str, n_iter: int = 15, cv: int = 3,
                    chronological: bool = False, seed: int = 42) -> TuneResult:
    cw = strategy == "class_weight"
    est = spec.builder(spec.defaults(), cw, pos_weight(y), seed)
    space = spec.search_space()
    sampler = make_sampler(strategy, y, seed)
    if sampler is not None:
        est = ImbPipeline([("sampler", sampler), ("clf", est)])
        space = {f"clf__{k}": v for k, v in space.items()}
    splitter = TimeSeriesSplit(cv) if chronological else StratifiedKFold(cv, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(est, space, n_iter=n_iter, scoring="average_precision", cv=splitter,
                                random_state=seed, n_jobs=1, error_score=np.nan, refit=False)
    search.fit(X, y)
    res = pd.DataFrame(search.cv_results_)
    if res["mean_test_score"].isna().all():
        raise ValueError("Every CV fold failed (too few frauds per fold?). Lower the number of folds.")
    res = res.sort_values("rank_test_score")
    strip = lambda d: {k.replace("clf__", ""): v for k, v in d.items()}
    best = strip(search.best_params_)
    table = pd.DataFrame([{**strip(p), "cv_pr_auc": s} for p, s in zip(res["params"], res["mean_test_score"])])
    return TuneResult({k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()},
                      float(search.best_score_), table)


def tune_unsupervised(spec: ModelSpec, X_train, y_train, X_val, y_val, n_iter: int = 10,
                      fit_on_legit_only: bool = False, seed: int = 42,
                      on_progress: Callable[[int, int], None] | None = None) -> TuneResult:
    rows = []
    cands = list(ParameterSampler(spec.search_space(), n_iter=n_iter, random_state=seed))
    for i, p in enumerate(cands):
        p = {k: (v.item() if hasattr(v, "item") else v) for k, v in p.items()}
        try:
            det = AnomalyDetector(spec.builder(p, False, 1.0, seed), fit_on_legit_only, seed=seed)
            det.fit(X_train, y_train)
            score = average_precision_score(y_val, det.score(X_val))
        except Exception:
            score = np.nan
        rows.append({**p, "val_pr_auc": score})
        if on_progress:
            on_progress(i + 1, len(cands))
    valid = [r for r in rows if not np.isnan(r["val_pr_auc"])]
    if not valid:
        raise ValueError("All candidate configurations failed.")
    best_row = max(valid, key=lambda r: r["val_pr_auc"])
    table = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).reset_index(drop=True)
    best = {k: v for k, v in best_row.items() if k != "val_pr_auc"}
    return TuneResult(best, float(best_row["val_pr_auc"]), table)
