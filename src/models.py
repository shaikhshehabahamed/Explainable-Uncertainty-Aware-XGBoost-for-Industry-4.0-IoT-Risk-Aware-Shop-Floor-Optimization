# src/models.py
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from .data import build_preprocessor, get_numeric_and_categorical_features
from .xgb_utils import make_xgb_common_params


TaskType = Literal["regression", "classification"]


class ToFloat32(TransformerMixin, BaseEstimator):
    """
    Cast transformer outputs to float32 while preserving sparse matrices.
    Ensures downstream XGBoost consumes float32 where possible.
    """

    def fit(self, X: Any, y: Any = None) -> "ToFloat32":
        return self

    def transform(self, X: Any) -> Union[np.ndarray, sparse.csr_matrix]:
        if sparse.issparse(X):
            X = X.tocsr()
            if X.dtype != np.float32:
                X = X.astype(np.float32)
            return X
        Xn = np.asarray(X)
        if Xn.dtype != np.float32:
            Xn = Xn.astype(np.float32, copy=False)
        return Xn


# -----------------------------
# Default training settings (can be overridden via CLI/env)
# -----------------------------
DEFAULT_SEED: int = int(os.environ.get("SEED", "42"))
DEFAULT_N_JOBS: int = int(os.environ.get("N_JOBS", "4"))
DEFAULT_MAX_BIN: int = int(os.environ.get("XGB_MAX_BIN", "256"))
DEFAULT_TREE_METHOD: str = str(os.environ.get("XGB_TREE_METHOD", "hist"))
DEFAULT_DEVICE: str = str(os.environ.get("XGB_DEVICE", "cpu"))

DEFAULT_REG_PARAMS: Dict[str, Any] = dict(
    n_estimators=int(os.environ.get("XGB_N_ESTIMATORS_REG", "1500")),
    learning_rate=float(os.environ.get("XGB_LEARNING_RATE", "0.03")),
    max_depth=int(os.environ.get("XGB_MAX_DEPTH_REG", "6")),
    min_child_weight=float(os.environ.get("XGB_MIN_CHILD_WEIGHT", "5")),
    subsample=float(os.environ.get("XGB_SUBSAMPLE", "0.8")),
    colsample_bytree=float(os.environ.get("XGB_COLSAMPLE_BYTREE", "0.8")),
    reg_lambda=float(os.environ.get("XGB_REG_LAMBDA", "2.0")),
    reg_alpha=float(os.environ.get("XGB_REG_ALPHA", "0.0")),
)

DEFAULT_CLF_PARAMS: Dict[str, Any] = dict(
    n_estimators=int(os.environ.get("XGB_N_ESTIMATORS_CLF", "2000")),
    learning_rate=float(os.environ.get("XGB_LEARNING_RATE", "0.03")),
    max_depth=int(os.environ.get("XGB_MAX_DEPTH_CLF", "4")),
    min_child_weight=float(os.environ.get("XGB_MIN_CHILD_WEIGHT", "5")),
    subsample=float(os.environ.get("XGB_SUBSAMPLE", "0.8")),
    colsample_bytree=float(os.environ.get("XGB_COLSAMPLE_BYTREE", "0.8")),
    reg_lambda=float(os.environ.get("XGB_REG_LAMBDA", "2.0")),
    reg_alpha=float(os.environ.get("XGB_REG_ALPHA", "0.0")),
    max_delta_step=float(os.environ.get("XGB_MAX_DELTA_STEP", "1.0")),
)


def get_reg_hpo_search_space(seed: int) -> Dict[str, Any]:
    """Random-search space for regression (used with sklearn ParameterSampler)."""
    rng = np.random.RandomState(int(seed))
    return {
        "learning_rate": rng.uniform(0.01, 0.08, size=1000),
        "max_depth": [3, 4, 5, 6, 7, 8],
        "min_child_weight": rng.uniform(1.0, 10.0, size=1000),
        "subsample": rng.uniform(0.6, 1.0, size=1000),
        "colsample_bytree": rng.uniform(0.6, 1.0, size=1000),
        "reg_lambda": rng.uniform(0.5, 6.0, size=1000),
        "reg_alpha": rng.uniform(0.0, 2.0, size=1000),
    }


def get_clf_hpo_search_space(seed: int) -> Dict[str, Any]:
    """Random-search space for classification (used with sklearn ParameterSampler)."""
    space = get_reg_hpo_search_space(seed)
    # Classification-specific knobs
    rng = np.random.RandomState(int(seed) + 123)
    space.update(
        {
            "max_delta_step": rng.uniform(0.0, 5.0, size=1000),
        }
    )
    return space


def make_regressor(
    *,
    device: str,
    seed: int,
    n_jobs: int,
    max_bin: int,
    tree_method: str,
    params: Optional[Dict[str, Any]] = None,
    objective: str = "reg:squarederror",
    eval_metric: str = "rmse",
) -> XGBRegressor:
    common = make_xgb_common_params(
        device=device, n_jobs=n_jobs, seed=seed, max_bin=max_bin, tree_method=tree_method, verbosity=0
    )
    p = dict(DEFAULT_REG_PARAMS)
    if params:
        p.update(params)
    return XGBRegressor(objective=objective, eval_metric=eval_metric, **common, **p)


def make_classifier(
    *,
    device: str,
    seed: int,
    n_jobs: int,
    max_bin: int,
    tree_method: str,
    scale_pos_weight: float,
    params: Optional[Dict[str, Any]] = None,
    objective: str = "binary:logistic",
    eval_metric: str = "logloss",
) -> XGBClassifier:
    common = make_xgb_common_params(
        device=device, n_jobs=n_jobs, seed=seed, max_bin=max_bin, tree_method=tree_method, verbosity=0
    )
    p = dict(DEFAULT_CLF_PARAMS)
    if params:
        p.update(params)
    return XGBClassifier(
        objective=objective,
        eval_metric=eval_metric,
        scale_pos_weight=float(scale_pos_weight),
        **common,
        **p,
    )


def build_pipeline_for_estimator(estimator: Any) -> Pipeline:
    """Build a preprocessing + float32 + estimator pipeline."""
    numeric_features, categorical_features = get_numeric_and_categorical_features()
    pre = build_preprocessor(numeric_features=numeric_features, categorical_features=categorical_features)
    return Pipeline(
        steps=[
            ("preprocess", pre),
            ("astype_float32", ToFloat32()),
            ("model", estimator),
        ]
    )