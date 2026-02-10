# src/data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------
# Spec-fixed dataset schema
# -----------------------------
EXPECTED_COLUMNS: List[str] = [
    "Unit_ID",
    "Process_Variable",
    "Setpoint",
    "PID_Output",
    "Fuzzy_Tuned_Kp",
    "Fuzzy_Tuned_Ki",
    "Fuzzy_Tuned_Kd",
    "Resource_Allocated",
    "Machine_Load",
    "Queue_Length",
    "Priority_Level",
    "Deadline",
    "Completion_Time",
    "Downtime",
    "Energy_Used",
    "Machine_Wear",
    "Temperature",
    "Vibration_Level",
    "Sensor_Status",
    "Predictive_Failure_Score",
    "Maintenance_Required",
    "Error",
    "Optimization_Success",
]

GROUP_COL = "Unit_ID"
SETPOINT_COL = "Setpoint"
SETPOINT_EXPECTED_VALUE = 80

CATEGORICAL_COLS: List[str] = ["Sensor_Status"]

# Targets are treated as outcomes and excluded from features (no leakage).
TARGET_COLS: List[str] = [
    "Completion_Time",
    "Downtime",
    "Energy_Used",
    "Machine_Wear",
    "Predictive_Failure_Score",
    "Maintenance_Required",
    "Error",
    "Optimization_Success",
]

# Classification targets (binary 0/1)
CLASSIFICATION_TARGETS: List[str] = ["Maintenance_Required", "Optimization_Success"]

# Regression targets (everything else in TARGET_COLS)
REGRESSION_TARGETS: List[str] = [t for t in TARGET_COLS if t not in CLASSIFICATION_TARGETS]

# Targets that are *probability-like* (bounded in [0,1]) but modeled as regression.
PROB_REGRESSION_TARGETS: List[str] = ["Predictive_Failure_Score"]

# Feature split exactly: context features + decision levers
CONTEXT_FEATURES: List[str] = [
    "Process_Variable",
    "Machine_Load",
    "Queue_Length",
    "Deadline",
    "Temperature",
    "Vibration_Level",
    "Sensor_Status",
]

DECISION_LEVERS: List[str] = [
    "PID_Output",
    "Fuzzy_Tuned_Kp",
    "Fuzzy_Tuned_Ki",
    "Fuzzy_Tuned_Kd",
    "Resource_Allocated",
    "Priority_Level",
]


@dataclass(frozen=True)
class RawSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame
    groups_train: pd.Series
    groups_test: pd.Series


def read_dataset(csv_path: str = "./distributed_manufacturing_dataset.csv") -> pd.DataFrame:
    """
    Read the dataset CSV.

    Resolution order:
      1) Use the provided csv_path (relative paths are resolved from the current working directory).
      2) Fallback to a dataset named 'distributed_manufacturing_dataset.csv' located in the repository root
         (i.e., the parent directory of the `src/` package).

    This makes the project portable across environments (no hard-coded /mnt/data dependency).
    """
    p = Path(csv_path)

    # Resolve relative paths from the current working directory
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()

    if not p.exists():
        fallback = (Path(__file__).resolve().parents[1] / "distributed_manufacturing_dataset.csv").resolve()
        if fallback.exists():
            p = fallback
        else:
            raise FileNotFoundError(
                f"Dataset not found at {p} and no fallback dataset found at {fallback}."
            )

    return pd.read_csv(p)

def validate_exact_columns(df: pd.DataFrame) -> None:
    """
    Strict schema check: the dataset must contain *exactly* EXPECTED_COLUMNS, including column order.

    This is best for research-grade reproducibility.

    For inference on "feature-only" files, use validate_required_columns(...).
    """
    expected = list(EXPECTED_COLUMNS)
    actual = list(df.columns.tolist())

    if actual == expected:
        return

    # Helpful diagnostics
    missing = [c for c in expected if c not in actual]
    unexpected = [c for c in actual if c not in expected]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if unexpected:
        raise ValueError(f"Unexpected columns present (schema must match exactly): {unexpected}")

    # Same set of columns, but different order
    raise ValueError(
        "Dataset columns contain the expected names but the order differs from EXPECTED_COLUMNS.\n"
        f"Expected order: {expected}\n"
        f"Actual order:   {actual}"
    )

def validate_required_columns(df: pd.DataFrame, required: List[str]) -> None:
    """Validation helper for inference: require a minimum set of columns."""
    missing = sorted(set(required) - set(df.columns.tolist()))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def drop_setpoint_constant(df: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
    """
    Drop Setpoint after validating it is constant 80.

    strict=True keeps paper/spec behavior (error if Setpoint missing or not constant).
    strict=False drops Setpoint if present (warns only if it is not constant). It does not error if missing.
    """
    if SETPOINT_COL not in df.columns:
            if strict:
                raise ValueError(f"Required column '{SETPOINT_COL}' is missing.")
            # Non-strict mode: allow feature-only inputs that omit Setpoint.
            return df.copy()

    nunique = df[SETPOINT_COL].nunique(dropna=False)
    first_val = df[SETPOINT_COL].iloc[0]

    if nunique != 1 or first_val != SETPOINT_EXPECTED_VALUE:
        msg = (
            f"'{SETPOINT_COL}' expected constant {SETPOINT_EXPECTED_VALUE} but observed "
            f"nunique={nunique}, first_value={first_val!r}."
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg + " Dropping Setpoint anyway because it is not used as a feature.", RuntimeWarning)

    # Drop in both strict and non-strict mode to keep downstream feature extraction consistent.
    return df.drop(columns=[SETPOINT_COL])

def get_feature_columns() -> Tuple[List[str], List[str], List[str]]:
    """Returns (context_features, decision_levers, all_features) with strict validation."""
    context = list(CONTEXT_FEATURES)
    levers = list(DECISION_LEVERS)
    all_features = context + levers

    overlap = sorted(set(context).intersection(levers))
    if overlap:
        raise ValueError(f"Context features and decision levers overlap: {overlap}")

    cat_missing = sorted(set(CATEGORICAL_COLS) - set(all_features))
    if cat_missing:
        raise ValueError(f"Categorical columns not included in features: {cat_missing}")

    leakage = sorted(set(TARGET_COLS).intersection(all_features))
    if leakage:
        raise ValueError(f"Targets must not be in features, but found: {leakage}")

    return context, levers, all_features


def split_features_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Returns:
      X_df: features (context + levers)
      y_df: targets
      groups: Unit_ID for group-aware split
    """
    if GROUP_COL not in df.columns:
        raise ValueError(f"Group column '{GROUP_COL}' missing.")

    missing_targets = sorted(set(TARGET_COLS) - set(df.columns))
    if missing_targets:
        raise ValueError(f"Missing required target columns: {missing_targets}")

    context, levers, all_features = get_feature_columns()

    missing_features = sorted(set(all_features) - set(df.columns))
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    X_df = df[all_features].copy()
    y_df = df[TARGET_COLS].copy()
    groups = df[GROUP_COL].copy()

    return X_df, y_df, groups


def extract_features_for_inference(df: pd.DataFrame, *, strict_setpoint: bool = False) -> pd.DataFrame:
    """
    Extract only the model features from an arbitrary CSV.

    - Allows extra columns (targets, identifiers, etc.)
    - Requires at least CONTEXT_FEATURES + DECISION_LEVERS
    - Drops Setpoint if present (strict_setpoint controls validation)
    """
    df2 = drop_setpoint_constant(df, strict=bool(strict_setpoint))
    _, _, all_features = get_feature_columns()
    validate_required_columns(df2, all_features)
    return df2[all_features].copy()


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """
    ColumnTransformer preprocessing:
      - Numeric: SimpleImputer(median) + StandardScaler
      - Categorical Sensor_Status: OneHotEncoder(handle_unknown="ignore")
    Notes:
      - OneHotEncoder API differs by sklearn version; we keep a compatibility fallback.
      - StandardScaler typically emits float64; we cast to float32 after transform for XGBoost.
    """
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # OneHotEncoder API differs by sklearn version; keep Windows/Anaconda-friendly.
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)

    cat_pipe = Pipeline(steps=[("onehot", ohe)])

    try:
        pre = ColumnTransformer(
            transformers=[
                ("num", num_pipe, numeric_features),
                ("cat", cat_pipe, categorical_features),
            ],
            remainder="drop",
            sparse_threshold=1.0,
            verbose_feature_names_out=False,
        )
    except TypeError:
        # sklearn<1.0 compatibility
        pre = ColumnTransformer(
            transformers=[
                ("num", num_pipe, numeric_features),
                ("cat", cat_pipe, categorical_features),
            ],
            remainder="drop",
            sparse_threshold=1.0,
        )
    return pre


def group_train_test_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    groups: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> RawSplit:
    """Group-aware split: GroupShuffleSplit test_size=0.2 random_state=42 on Unit_ID groups."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    (train_idx, test_idx) = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    groups_train = groups.iloc[train_idx].reset_index(drop=True)
    groups_test = groups.iloc[test_idx].reset_index(drop=True)

    return RawSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        groups_train=groups_train,
        groups_test=groups_test,
    )


def get_numeric_and_categorical_features() -> Tuple[List[str], List[str]]:
    context, levers, all_features = get_feature_columns()
    categorical = list(CATEGORICAL_COLS)
    numeric = [c for c in all_features if c not in categorical]
    return numeric, categorical