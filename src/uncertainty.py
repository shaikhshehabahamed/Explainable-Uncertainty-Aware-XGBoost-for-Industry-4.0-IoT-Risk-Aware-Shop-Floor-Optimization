# src/uncertainty.py
from __future__ import annotations

"""
Uncertainty helpers (dependency-light):

Regression:
- Group-aware OOF conformal residual quantile (caller provides OOF predictions).
- Returns symmetric prediction intervals: [pred - q, pred + q] on the *training scale*.
- For probability-like regression (e.g., Predictive_Failure_Score), we recommend computing
  q in the transformed space (logit) and inverse-transforming bounds (sigmoid) to keep
  predictions in [0,1].

Classification:
- Isotonic calibration on OOF probabilities.
- Conservative bounds via bin-wise Wilson confidence intervals over calibrated OOF probabilities.

This module is intentionally standalone and avoids requiring optional packages like MAPIE.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np


# -----------------------------
# Conformal regression utilities
# -----------------------------
def conformal_quantile(abs_residuals: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile for nonnegative residuals.

    Uses the standard conformal correction:
      q_level = ceil((n + 1) * (1 - alpha)) / n
    and takes the 'higher' empirical quantile.
    """
    r = np.asarray(abs_residuals, dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n <= 0:
        return 0.0
    q_level = float(np.ceil((n + 1) * (1.0 - float(alpha))) / n)
    q_level = min(1.0, max(0.0, q_level))
    try:
        q = float(np.quantile(r, q_level, method="higher"))
    except TypeError:  # numpy<1.22
        q = float(np.quantile(r, q_level, interpolation="higher"))
    return 0.0 if not np.isfinite(q) else q


def prediction_interval(pred: np.ndarray, q: float) -> Tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float32).ravel()
    qf = float(q)
    lo = (pred - qf).astype(np.float32, copy=False)
    hi = (pred + qf).astype(np.float32, copy=False)
    return lo, hi


# -----------------------------
# Classification bounds utilities
# -----------------------------
def wilson_interval(k: int, n: int, alpha: float = 0.1) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return 0.0, 1.0
    z = float(_z_for_two_sided_alpha(alpha))
    phat = float(k) / float(n)
    denom = 1.0 + (z * z) / float(n)
    centre = (phat + (z * z) / (2.0 * float(n))) / denom
    half = (z / denom) * np.sqrt((phat * (1.0 - phat) / float(n)) + (z * z) / (4.0 * float(n) * float(n)))
    return max(0.0, centre - half), min(1.0, centre + half)


def _z_for_two_sided_alpha(alpha: float) -> float:
    """Approximate z for a two-sided normal interval.

    We avoid scipy.stats as a hard dependency here. The approximation is accurate enough
    for confidence bounds used as conservative constraints.
    """
    a = float(alpha)
    a = min(0.999999, max(1e-12, a))
    # For two-sided alpha, tail probability is alpha/2
    p = 1.0 - a / 2.0
    # Inverse CDF approximation (Peter John Acklam).
    return float(_norm_ppf(p))


def _norm_ppf(p: float) -> float:
    # Acklam's approximation: https://web.archive.org/web/20150910044729/http://home.online.no/~pjacklam/notes/invnorm/
    # Coefficients
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    # Define break-points
    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = np.sqrt(-2*np.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if phigh < p:
        q = np.sqrt(-2*np.log(1-p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)


@dataclass(frozen=True)
class ProbabilityBinBounds:
    edges: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    alpha: float

    def lookup(self, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return lookup_bin_bounds(p, self.edges, self.lower, self.upper)


def fit_probability_bounds_from_calibration(
    p_cal: np.ndarray,
    y_true: np.ndarray,
    *,
    alpha: float = 0.1,
    n_bins: int = 10,
) -> ProbabilityBinBounds:
    """Fit bin-wise Wilson bounds as a function of calibrated probability p_cal."""
    p = np.asarray(p_cal, dtype=np.float64).ravel()
    y = np.asarray(y_true, dtype=int).ravel()
    if p.size != y.size:
        raise ValueError("p_cal and y_true must have the same length.")
    if p.size == 0:
        raise ValueError("Empty calibration arrays.")

    # Bin edges evenly in probability space
    n_bins = int(max(2, n_bins))
    edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
    lower = np.zeros((n_bins,), dtype=np.float64)
    upper = np.ones((n_bins,), dtype=np.float64)

    # Assign samples to bins
    bin_ids = np.digitize(p, edges[1:-1], right=False)  # 0..n_bins-1

    for b in range(n_bins):
        mask = bin_ids == b
        n = int(np.sum(mask))
        if n <= 0:
            lower[b], upper[b] = 0.0, 1.0
            continue
        k = int(np.sum(y[mask] == 1))
        lb, ub = wilson_interval(k, n, alpha=float(alpha))
        lower[b], upper[b] = lb, ub

    return ProbabilityBinBounds(
        edges=edges.astype(np.float32),
        lower=lower.astype(np.float32),
        upper=upper.astype(np.float32),
        alpha=float(alpha),
    )


def lookup_bin_bounds(
    p_cal: np.ndarray,
    edges: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lookup per-sample (lower, upper) bounds based on calibrated probability bins."""
    p = np.asarray(p_cal, dtype=np.float64).ravel()
    edges = np.asarray(edges, dtype=np.float64).ravel()
    lower = np.asarray(lower, dtype=np.float64).ravel()
    upper = np.asarray(upper, dtype=np.float64).ravel()

    if edges.size < 2:
        raise ValueError("edges must have length >= 2.")
    n_bins = edges.size - 1
    if lower.size != n_bins or upper.size != n_bins:
        raise ValueError("lower/upper must have length len(edges)-1.")

    # digitize: returns 0..n_bins-1
    bin_ids = np.digitize(p, edges[1:-1], right=False)

    lb = lower[bin_ids]
    ub = upper[bin_ids]
    return lb.astype(np.float32), ub.astype(np.float32)