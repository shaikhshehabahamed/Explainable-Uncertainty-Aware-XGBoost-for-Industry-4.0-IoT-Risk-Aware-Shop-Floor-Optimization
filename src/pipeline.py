# src/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer

from .data import (
    CONTEXT_FEATURES,
    DECISION_LEVERS,
    TARGET_COLS,
    build_preprocessor,
    drop_setpoint_constant,
    get_numeric_and_categorical_features,
    group_train_test_split,
    read_dataset,
    split_features_targets,
    validate_exact_columns,
)


@dataclass(frozen=True)
class PreparedData:
    # Raw split
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame

    # Groups (Unit_ID) aligned to the raw splits
    groups_train: pd.Series
    groups_test: pd.Series

    # Feature split: context + levers (raw)
    X_train_context: pd.DataFrame
    X_test_context: pd.DataFrame
    X_train_levers: pd.DataFrame
    X_test_levers: pd.DataFrame

    # Preprocessed matrices (float32, sparse CSR preferred)
    X_train: Union[np.ndarray, sparse.csr_matrix]
    X_test: Union[np.ndarray, sparse.csr_matrix]

    # Preprocessor + feature names
    preprocessor: ColumnTransformer
    feature_names_out: List[str]


def _to_float32_matrix(X: Any) -> Union[np.ndarray, sparse.csr_matrix]:
    """Ensure outputs are float32 and CSR when sparse (GPU/CPU friendly)."""
    if sparse.issparse(X):
        X = X.tocsr()
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        return X
    Xn = np.asarray(X)
    if Xn.dtype != np.float32:
        Xn = Xn.astype(np.float32, copy=False)
    return Xn


def prepare_data(
    csv_path: str = "./distributed_manufacturing_dataset.csv",
    *,
    strict_schema: bool = True,
    seed: int = 42,
) -> PreparedData:
    """
    End-to-end preparation:
      - Read dataset
      - (Optional) enforce exact columns
      - Drop Setpoint (constant 80)
      - Feature split exactly: context features + decision levers
      - Group-aware split: GroupShuffleSplit test_size=0.2 random_state=seed on Unit_ID groups
      - Preprocess with ColumnTransformer:
          Numeric: SimpleImputer(median) + StandardScaler
          Categorical Sensor_Status: OneHotEncoder(handle_unknown="ignore")
      - Transform to float32 (sparse CSR when applicable)

    strict_schema=True keeps the research/paper behavior (exact schema required).
    """
    df = read_dataset(csv_path)
    if strict_schema:
        validate_exact_columns(df)

    df = drop_setpoint_constant(df, strict=bool(strict_schema))

    X_df, y_df, groups = split_features_targets(df)

    # Strictly enforce that raw features match context+levers exactly (no drift)
    expected_features = set(CONTEXT_FEATURES + DECISION_LEVERS)
    actual_features = set(X_df.columns.tolist())
    if actual_features != expected_features:
        missing = sorted(expected_features - actual_features)
        extra = sorted(actual_features - expected_features)
        raise ValueError(
            "Feature columns must be exactly CONTEXT_FEATURES + DECISION_LEVERS.\n"
            f"Missing: {missing}\nExtra: {extra}"
        )

    # Ensure targets are exactly as required
    if set(y_df.columns.tolist()) != set(TARGET_COLS):
        raise ValueError(f"Targets must be exactly: {TARGET_COLS}\nObserved: {y_df.columns.tolist()}")

    raw = group_train_test_split(X_df, y_df, groups, test_size=0.2, random_state=int(seed))

    # Split raw features into context + levers
    X_train_context = raw.X_train[CONTEXT_FEATURES].copy()
    X_test_context = raw.X_test[CONTEXT_FEATURES].copy()
    X_train_levers = raw.X_train[DECISION_LEVERS].copy()
    X_test_levers = raw.X_test[DECISION_LEVERS].copy()

    # Build and fit preprocessing on TRAIN split (for convenience outputs only).
    # Note: for CV/OoF training we re-fit preprocessing inside folds to prevent leakage.
    numeric_features, categorical_features = get_numeric_and_categorical_features()
    pre = build_preprocessor(numeric_features=numeric_features, categorical_features=categorical_features)

    X_train_proc = pre.fit_transform(raw.X_train)
    X_test_proc = pre.transform(raw.X_test)

    X_train_proc = _to_float32_matrix(X_train_proc)
    X_test_proc = _to_float32_matrix(X_test_proc)

    # Feature names out (best effort, sklearn version dependent)
    try:
        feature_names = pre.get_feature_names_out().tolist()
    except Exception:
        n_out = int(X_train_proc.shape[1])
        feature_names = [f"f{i}" for i in range(n_out)]

    return PreparedData(
        X_train_raw=raw.X_train,
        X_test_raw=raw.X_test,
        y_train=raw.y_train,
        y_test=raw.y_test,
        groups_train=raw.groups_train,
        groups_test=raw.groups_test,
        X_train_context=X_train_context,
        X_test_context=X_test_context,
        X_train_levers=X_train_levers,
        X_test_levers=X_test_levers,
        X_train=X_train_proc,
        X_test=X_test_proc,
        preprocessor=pre,
        feature_names_out=feature_names,
    )


if __name__ == "__main__":
    data = prepare_data("./distributed_manufacturing_dataset.csv", strict_schema=False)
    Xtr = data.X_train
    Xte = data.X_test

    def _shape(x: Any) -> Tuple[int, int]:
        return (x.shape[0], x.shape[1])

    print("Prepared data:")
    print(f"  X_train: {_shape(Xtr)} dtype={getattr(Xtr,'dtype',None)} sparse={sparse.issparse(Xtr)}")
    print(f"  X_test : {_shape(Xte)} dtype={getattr(Xte,'dtype',None)} sparse={sparse.issparse(Xte)}")
    print(f"  y_train: {data.y_train.shape} columns={list(data.y_train.columns)}")
    print(f"  y_test : {data.y_test.shape} columns={list(data.y_test.columns)}")
    print(f"  feature_names_out: {len(data.feature_names_out)}")