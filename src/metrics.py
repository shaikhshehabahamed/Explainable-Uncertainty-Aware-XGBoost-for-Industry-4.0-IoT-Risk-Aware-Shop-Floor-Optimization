# src/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def validate_binary_labels(y: Any, target: str) -> np.ndarray:
    """Validate and coerce binary labels to int {0,1}."""
    arr = np.asarray(y)
    if pd.isna(arr).any():
        raise ValueError(f"Classification target '{target}' contains NaN labels; please clean/drop these rows.")

    uniq = np.unique(arr)
    if uniq.size == 0:
        raise ValueError(f"Target '{target}' has no labels.")
    if not set(uniq.tolist()).issubset({0, 1, False, True, 0.0, 1.0}):
        raise ValueError(f"Classification target '{target}' must be binary in {{0,1}}; found values: {uniq.tolist()}")
    return arr.astype(int)


def compute_scale_pos_weight(y: Any) -> float:
    y_arr = np.asarray(y).astype(int)
    num_pos = int(np.sum(y_arr == 1))
    num_neg = int(np.sum(y_arr == 0))
    return float(num_neg) / float(max(1, num_pos))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"rmse": rmse, "mae": mae}


def classification_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int).ravel()
    p = np.asarray(y_proba, dtype=float).ravel()
    # Numerical safety: avoid exactly 0/1 probabilities (logloss can overflow)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    thr = float(threshold)

    y_pred = (p >= thr).astype(int)

    out: Dict[str, float] = {
        "threshold": thr,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }

    # Log loss can fail on degenerate y; pass labels to be safe.
    try:
        out["logloss"] = float(log_loss(y_true, p, labels=[0, 1]))
    except Exception:
        out["logloss"] = float("nan")

    # AUC metrics undefined if only one class present
    if len(np.unique(y_true)) == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, p))
        except Exception:
            out["roc_auc"] = float("nan")
        try:
            out["pr_auc"] = float(average_precision_score(y_true, p))
        except Exception:
            out["pr_auc"] = float("nan")
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")

    return out


def select_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    metric: str = "f1",
    grid: Optional[np.ndarray] = None,
) -> float:
    """Select a decision threshold that optimizes a metric on provided data.

    This is intended to be used on *OOF calibrated probabilities* to avoid leakage.

    metric choices:
      - "f1" (default)
      - "balanced_accuracy"
      - "youden" (TPR - FPR)
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    p = np.asarray(y_proba, dtype=float).ravel()
    # Numerical safety: avoid exactly 0/1 probabilities (logloss can overflow)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)

    if grid is None:
        # A practical grid dense enough for thresholding.
        grid = np.linspace(0.01, 0.99, 99)

    best_thr = 0.5
    best_score = -1e18

    for thr in grid:
        y_pred = (p >= thr).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, y_pred)
        elif metric == "youden":
            # youden = TPR - FPR
            tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            tn = int(np.sum((y_true == 0) & (y_pred == 0)))
            tpr = tp / max(1, (tp + fn))
            fpr = fp / max(1, (fp + tn))
            score = tpr - fpr
        else:
            raise ValueError(f"Unknown threshold metric: {metric}")

        if score > best_score:
            best_score = float(score)
            best_thr = float(thr)

    return best_thr