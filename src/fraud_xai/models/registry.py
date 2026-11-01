"""Model catalogue. Each ModelSpec declares its hyper-parameters ONCE; the UI form,
the random-search space and the builder are all derived from it.
Add a model = add one ModelSpec here."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier

from fraud_xai.models.unsupervised import DBSCANDetector


@dataclass(frozen=True)
class Param:
    name: str
    kind: str                      # int | float | choice | bool
    default: Any
    low: Any = None
    high: Any = None
    choices: tuple = ()
    log: bool = False
    help: str = ""

    def distribution(self):
        if self.kind == "int":
            return randint(self.low, self.high + 1)
        if self.kind == "float":
            return loguniform(self.low, self.high) if self.log else uniform(self.low, self.high - self.low)
        if self.kind == "choice":
            return list(self.choices)
        return [True, False]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    family: str                    # supervised | unsupervised
    params: tuple[Param, ...]
    builder: Callable[..., Any]    # (params, class_weighted, pos_weight, seed) -> estimator
    explainer: str                 # tree | linear | permutation
    description: str = ""

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params}

    def search_space(self) -> dict:
        return {p.name: p.distribution() for p in self.params}


# ---- supervised ------------------------------------------------------------
def _logreg(p, cw, pw, seed):
    return LogisticRegression(**p, max_iter=2000, class_weight="balanced" if cw else None, random_state=seed)

def _rf(p, cw, pw, seed):
    return RandomForestClassifier(**p, class_weight="balanced" if cw else None, n_jobs=-1, random_state=seed)

def _xgb(p, cw, pw, seed):
    return XGBClassifier(**p, scale_pos_weight=pw if cw else 1.0, eval_metric="aucpr",
                         tree_method="hist", n_jobs=-1, random_state=seed)

def _hgb(p, cw, pw, seed):
    return HistGradientBoostingClassifier(**p, class_weight="balanced" if cw else None, random_state=seed)

# ---- unsupervised ----------------------------------------------------------
def _iforest(p, cw, pw, seed):
    return IsolationForest(**p, n_jobs=-1, random_state=seed)

def _dbscan(p, cw, pw, seed):
    return DBSCANDetector(**p)

def _lof(p, cw, pw, seed):
    return LocalOutlierFactor(**p, novelty=True, n_jobs=-1)

def _ocsvm(p, cw, pw, seed):
    return OneClassSVM(**p, kernel="rbf")


REGISTRY: dict[str, ModelSpec] = {s.key: s for s in [
    ModelSpec("logreg", "Logistic Regression", "supervised", (
        Param("C", "float", 1.0, 1e-3, 1e2, log=True, help="Inverse regularisation strength"),
    ), _logreg, "linear", "Interpretable linear baseline."),
    ModelSpec("rf", "Random Forest", "supervised", (
        Param("n_estimators", "int", 200, 50, 500),
        Param("max_depth", "int", 12, 3, 30),
        Param("min_samples_leaf", "int", 2, 1, 20),
        Param("max_features", "choice", "sqrt", choices=("sqrt", "log2")),
    ), _rf, "tree", "Robust bagging ensemble."),
    ModelSpec("xgboost", "XGBoost", "supervised", (
        Param("n_estimators", "int", 300, 50, 600),
        Param("max_depth", "int", 6, 2, 10),
        Param("learning_rate", "float", 0.1, 0.01, 0.3, log=True),
        Param("subsample", "float", 0.8, 0.5, 1.0),
        Param("colsample_bytree", "float", 0.8, 0.5, 1.0),
    ), _xgb, "tree", "Gradient boosting - usually the strongest tabular baseline."),
    ModelSpec("hgb", "HistGradientBoosting", "supervised", (
        Param("max_iter", "int", 200, 50, 500),
        Param("learning_rate", "float", 0.1, 0.01, 0.3, log=True),
        Param("max_depth", "int", 6, 2, 12),
        Param("l2_regularization", "float", 0.1, 1e-3, 10.0, log=True),
    ), _hgb, "permutation", "scikit-learn native boosting (fast on large data)."),
    ModelSpec("iforest", "Isolation Forest", "unsupervised", (
        Param("n_estimators", "int", 200, 50, 500),
        Param("max_samples", "int", 256, 64, 2048),
        Param("max_features", "float", 1.0, 0.3, 1.0),
    ), _iforest, "permutation", "Isolates anomalies with random splits (short paths = anomalous)."),
    ModelSpec("dbscan", "DBSCAN (distance to core points)", "unsupervised", (
        Param("min_samples", "int", 10, 3, 50, help="Neighbours needed for a core point"),
        Param("eps_quantile", "float", 0.9, 0.5, 0.99, help="eps = this quantile of the k-distance curve"),
    ), _dbscan, "permutation", "Density clustering; far from any dense region = suspicious."),
    ModelSpec("lof", "Local Outlier Factor", "unsupervised", (
        Param("n_neighbors", "int", 20, 5, 100),
    ), _lof, "permutation", "Compares local density with neighbours' density."),
    ModelSpec("ocsvm", "One-Class SVM", "unsupervised", (
        Param("nu", "float", 0.05, 0.001, 0.5, log=True, help="Upper bound on outlier fraction"),
        Param("gamma", "choice", "scale", choices=("scale", "auto")),
    ), _ocsvm, "permutation", "Learns a boundary around normal behaviour (sub-sampled for speed)."),
]}


def specs_by_family(family: str) -> list[ModelSpec]:
    return [s for s in REGISTRY.values() if s.family == family]
