"""Use-cases orchestrating data -> features -> split -> preprocessing -> training.
Streamlit pages only call these functions (no ML logic in the UI)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fraud_xai.config import RANDOM_STATE
from fraud_xai.data.schema import Schema, binarize_target, parse_time
from fraud_xai.data.splitter import Split, split_dataset
from fraud_xai.evaluation.metrics import ranking_metrics
from fraud_xai.features.engineering import FeatureConfig, engineer_features
from fraud_xai.features.preprocessing import TabularPreprocessor
from fraud_xai.models.base import AnomalyDetector, ProbaDetector
from fraud_xai.models.imbalance import make_sampler, pos_weight
from fraud_xai.models.registry import REGISTRY
from fraud_xai.models.tuning import TuneResult, tune_supervised, tune_unsupervised


@dataclass
class PrepConfig:
    use_temporal: bool = True
    use_behavioural: bool = True
    scaler: str = "power"
    corr_threshold: float = 0.95
    max_onehot: int = 10
    seed: int = RANDOM_STATE


@dataclass
class Prepared:
    schema: Schema
    cfg: PrepConfig
    feature_cfg: FeatureConfig
    split: Split
    preprocessor: TabularPreprocessor
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    amounts: dict = field(default_factory=dict)       # split name -> np.ndarray | None
    raw: dict = field(default_factory=dict)           # split name -> engineered raw features (readable)

    @property
    def y(self) -> dict:
        return {"train": self.split.y_train, "val": self.split.y_val, "test": self.split.y_test}

    @property
    def X(self) -> dict:
        return {"train": self.X_train, "val": self.X_val, "test": self.X_test}


def prepare(df: pd.DataFrame, schema: Schema, cfg: PrepConfig) -> Prepared:
    y = binarize_target(df[schema.target], schema.positive_label).reset_index(drop=True)
    df = df.reset_index(drop=True)
    time_s = parse_time(df[schema.time_col]) if schema.time_col else None
    fcfg = FeatureConfig(schema.time_col, schema.amount_col, schema.entity_col,
                         cfg.use_temporal, cfg.use_behavioural)

    keep = [c for c in df.columns if c != schema.target and c not in schema.drop_cols]
    for c in (schema.time_col, schema.entity_col, schema.amount_col):  # needed for features
        if c and c not in keep:
            keep.append(c)
    X_eng = engineer_features(df[keep], fcfg)
    amount = (pd.to_numeric(df[schema.amount_col], errors="coerce").fillna(0).to_numpy()
              if schema.amount_col else None)
    X_eng["__amount__"] = amount if amount is not None else 0.0   # carried through the split

    split = split_dataset(X_eng, y, time_s, seed=cfg.seed)
    amounts = {}
    for name in ("train", "val", "test"):
        Xs = getattr(split, f"X_{name}")
        amounts[name] = Xs["__amount__"].to_numpy() if amount is not None else None
        setattr(split, f"X_{name}", Xs.drop(columns="__amount__"))

    pre = TabularPreprocessor(cfg.scaler, cfg.corr_threshold, cfg.max_onehot).fit(split.X_train)
    return Prepared(schema, cfg, fcfg, split, pre,
                    pre.transform(split.X_train), pre.transform(split.X_val), pre.transform(split.X_test),
                    amounts, {"train": split.X_train, "val": split.X_val, "test": split.X_test})


@dataclass
class TrainedModel:
    key: str
    label: str
    family: str
    params: dict
    imbalance: str
    detector: object
    scores: dict                      # split name -> np.ndarray
    metrics: dict                     # split name -> {pr_auc, roc_auc}
    fit_seconds: float
    tuning: TuneResult | None = None


def _fit(prepared: Prepared, key: str, params: dict, strategy: str, fit_legit_only: bool, seed: int):
    spec = REGISTRY[key]
    Xtr, ytr = prepared.X_train, prepared.split.y_train
    if spec.family == "supervised":
        sampler = make_sampler(strategy, ytr, seed)
        Xf, yf = (Xtr, ytr) if sampler is None else sampler.fit_resample(Xtr, ytr)   # TRAIN ONLY
        est = spec.builder(params, strategy == "class_weight", pos_weight(ytr), seed)
        est.fit(Xf, yf)
        return ProbaDetector(est)
    return AnomalyDetector(spec.builder(params, False, 1.0, seed), fit_legit_only, seed=seed).fit(Xtr, ytr)


def train_model(prepared: Prepared, key: str, params: dict | None = None, strategy: str = "none",
                fit_legit_only: bool = False, tuning: TuneResult | None = None,
                seed: int = RANDOM_STATE) -> TrainedModel:
    spec = REGISTRY[key]
    params = {**spec.defaults(), **(params or {})}
    t0 = time.time()
    det = _fit(prepared, key, params, strategy, fit_legit_only, seed)
    scores = {n: det.score(prepared.X[n]) for n in ("train", "val", "test")}
    metrics = {n: ranking_metrics(prepared.y[n], scores[n]) for n in ("val", "test")}
    return TrainedModel(key, spec.label, spec.family, params,
                        strategy if spec.family == "supervised" else "n/a",
                        det, scores, metrics, time.time() - t0, tuning)


def tune_model(prepared: Prepared, key: str, strategy: str, n_iter: int, cv: int,
               fit_legit_only: bool = False, seed: int = RANDOM_STATE, on_progress=None) -> TuneResult:
    spec = REGISTRY[key]
    if spec.family == "supervised":
        return tune_supervised(spec, prepared.X_train, prepared.split.y_train, strategy, n_iter, cv,
                               prepared.split.chronological, seed)
    return tune_unsupervised(spec, prepared.X_train, prepared.split.y_train, prepared.X_val,
                             prepared.split.y_val, n_iter, fit_legit_only, seed, on_progress)


def compare_imbalance(prepared: Prepared, key: str, params: dict, strategies: list[str],
                      seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Justify the imbalance strategy by VALIDATION PR-AUC (test untouched)."""
    rows = []
    for s in strategies:
        det = _fit(prepared, key, {**REGISTRY[key].defaults(), **params}, s, False, seed)
        m = ranking_metrics(prepared.split.y_val, det.score(prepared.X_val))
        rows.append({"strategy": s, "val_pr_auc": m["pr_auc"], "val_roc_auc": m["roc_auc"]})
    return pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).reset_index(drop=True)
