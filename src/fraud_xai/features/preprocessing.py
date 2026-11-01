"""Leak-free tabular preprocessing. EVERYTHING is fitted on the training split
only: medians, categories, constant/correlated column removal and scaling.
(The original notebook imputed with the full-dataset median.)"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler

SCALERS = {"power": "Yeo-Johnson (PowerTransformer)", "standard": "StandardScaler",
           "robust": "RobustScaler", "none": "No scaling"}


class TabularPreprocessor:
    def __init__(self, scaler: str = "power", corr_threshold: float = 0.95,
                 max_onehot: int = 10):
        self.scaler, self.corr_threshold, self.max_onehot = scaler, corr_threshold, max_onehot

    # -- internals -------------------------------------------------------
    def _matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        parts = {}
        for c in self.num_cols_:
            s = pd.to_numeric(X[c], errors="coerce") if c in X else pd.Series(np.nan, index=X.index)
            parts[c] = s.astype(float).replace([np.inf, -np.inf], np.nan).fillna(self.medians_[c])
        for c, levels in self.cat_levels_.items():
            s = X[c].astype(str).where(X[c].notna(), "missing") if c in X else pd.Series("missing", index=X.index)
            for lv in levels:
                parts[f"{c}={lv}"] = (s == lv).astype(float)
        return pd.DataFrame(parts, index=X.index)

    def fit(self, X: pd.DataFrame) -> "TabularPreprocessor":
        self.input_columns_ = list(X.columns)
        is_num = [c for c in X if pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])]
        others = [c for c in X if c not in is_num]
        self.num_cols_ = is_num
        self.cat_levels_, high_card = {}, []
        for c in others:
            if X[c].nunique() <= self.max_onehot:
                self.cat_levels_[c] = sorted(X[c].astype(str).where(X[c].notna(), "missing").unique())
            else:
                high_card.append(c)
        self.medians_ = {c: float(pd.to_numeric(X[c], errors="coerce")
                                  .replace([np.inf, -np.inf], np.nan).median()) for c in is_num}
        self.medians_ = {c: (0.0 if np.isnan(v) else v) for c, v in self.medians_.items()}

        M = self._matrix(X)
        const = [c for c in M if M[c].std() == 0]
        M = M.drop(columns=const)
        sample = M.sample(min(len(M), 50_000), random_state=0) if len(M) > 50_000 else M
        corr = sample.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, bool), k=1))
        corr_drop = [c for c in upper if (upper[c] > self.corr_threshold).any()]
        M = M.drop(columns=corr_drop)
        self.feature_names_ = list(M.columns)

        self._scaler = {"power": lambda: PowerTransformer(method="yeo-johnson"),
                        "standard": StandardScaler, "robust": RobustScaler}.get(self.scaler, lambda: None)()
        if self._scaler is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._scaler.fit(M.values)
        self.report_ = {"input_columns": len(self.input_columns_),
                        "output_features": len(self.feature_names_),
                        "dropped_high_cardinality": high_card, "dropped_constant": const,
                        "dropped_correlated": corr_drop}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        M = self._matrix(X).reindex(columns=self.feature_names_, fill_value=0.0)
        if self._scaler is None:
            return M
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arr = self._scaler.transform(M.values)
        return pd.DataFrame(np.nan_to_num(arr), index=M.index, columns=self.feature_names_)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
