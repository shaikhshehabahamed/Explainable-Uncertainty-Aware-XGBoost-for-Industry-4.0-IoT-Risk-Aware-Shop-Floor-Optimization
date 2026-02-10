# src/transforms.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


TransformName = Literal["identity", "logit"]


def expit(x: np.ndarray) -> np.ndarray:
    """Numerically-stable sigmoid."""
    x = np.asarray(x, dtype=np.float64)
    # Avoid overflow:
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out.astype(np.float32)


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Inverse sigmoid with clipping to avoid infinities."""
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, float(eps), 1.0 - float(eps))
    out = np.log(p / (1.0 - p))
    return out.astype(np.float32)


@dataclass(frozen=True)
class TargetTransform:
    name: TransformName = "identity"
    eps: float = 1e-6

    def forward(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).ravel()
        if self.name == "identity":
            return y.astype(np.float32)
        if self.name == "logit":
            return logit(y, eps=self.eps)
        raise ValueError(f"Unknown transform: {self.name}")

    def inverse(self, y_t: np.ndarray) -> np.ndarray:
        y_t = np.asarray(y_t, dtype=np.float64).ravel()
        if self.name == "identity":
            return y_t.astype(np.float32)
        if self.name == "logit":
            return expit(y_t)
        raise ValueError(f"Unknown transform: {self.name}")