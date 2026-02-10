# src/artifacts.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from joblib import dump, load
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline

from .transforms import TargetTransform
from .uncertainty import lookup_bin_bounds, prediction_interval, ProbabilityBinBounds


@dataclass
class RegressionArtifact:
    target: str
    pipeline: Pipeline
    abs_resid_q: float
    alpha: float = 0.1
    transform: TargetTransform = field(default_factory=TargetTransform)

    def predict_mean_and_bounds(self, X_df) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (mean, lower, upper) on the original target scale."""
        pred_t = np.asarray(self.pipeline.predict(X_df), dtype=np.float32).ravel()
        lo_t, hi_t = prediction_interval(pred_t, self.abs_resid_q)

        mean = self.transform.inverse(pred_t)
        lo = self.transform.inverse(lo_t)
        hi = self.transform.inverse(hi_t)

        # Ensure monotonic ordering even under numerical jitter.
        lo2 = np.minimum(lo, hi)
        hi2 = np.maximum(lo, hi)
        mean2 = np.clip(mean, lo2, hi2)
        return mean2.astype(np.float32), lo2.astype(np.float32), hi2.astype(np.float32)


@dataclass
class ClassificationArtifact:
    target: str
    pipeline: Pipeline
    isotonic: Optional[IsotonicRegression] = None
    bounds: Optional[ProbabilityBinBounds] = None
    decision_threshold: float = 0.5

    def predict_proba_raw(self, X_df) -> np.ndarray:
        p = self.pipeline.predict_proba(X_df)[:, 1]
        return np.asarray(p, dtype=np.float32).ravel()

    def predict_proba_calibrated(self, X_df) -> np.ndarray:
        p_raw = self.predict_proba_raw(X_df)
        if self.isotonic is None:
            return np.clip(p_raw, 0.0, 1.0)
        p_cal = self.isotonic.predict(p_raw)
        return np.clip(np.asarray(p_cal, dtype=np.float32).ravel(), 0.0, 1.0)

    def predict_bounds(self, X_df) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        p_cal = self.predict_proba_calibrated(X_df)
        if self.bounds is None:
            # No bounds available => degenerate bounds (not heuristic)
            return p_cal, p_cal, p_cal
        lb, ub = self.bounds.lookup(p_cal)
        return p_cal.astype(np.float32), lb.astype(np.float32), ub.astype(np.float32)

    def predict_label(self, X_df, *, threshold: Optional[float] = None) -> np.ndarray:
        thr = float(self.decision_threshold if threshold is None else threshold)
        p = self.predict_proba_calibrated(X_df)
        return (p >= thr).astype(int)


@dataclass
class ModelBundle:
    """Convenience container for all targets + metadata."""
    regression: Dict[str, RegressionArtifact]
    classification: Dict[str, ClassificationArtifact]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        dump(self, str(p))
        return str(p)

    @staticmethod
    def load(path: str) -> "ModelBundle":
        obj = load(path)
        if not isinstance(obj, ModelBundle):
            raise TypeError(f"Loaded object is not a ModelBundle: {type(obj)}")
        return obj