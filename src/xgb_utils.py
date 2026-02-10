# src/xgb_utils.py
from __future__ import annotations

import inspect
import os
import subprocess
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def xgb_best_iteration_plus_one(model: Any) -> Optional[int]:
    """Return best_iteration+1 (n_estimators to use) if present, else None."""
    bi = getattr(model, "best_iteration", None)
    return _safe_int(bi + 1 if bi is not None else None)


def xgb_predict_best(model: Any, X: Any) -> np.ndarray:
    """Predict using best_iteration when early stopping was used (XGBoost sklearn API)."""
    best_n = xgb_best_iteration_plus_one(model)
    if best_n is None:
        return np.asarray(model.predict(X), dtype=np.float32).ravel()

    try:
        return np.asarray(model.predict(X, iteration_range=(0, best_n)), dtype=np.float32).ravel()
    except TypeError:
        try:
            return np.asarray(model.predict(X, ntree_limit=best_n), dtype=np.float32).ravel()
        except TypeError:
            return np.asarray(model.predict(X), dtype=np.float32).ravel()


def xgb_predict_proba_best(model: Any, X: Any) -> np.ndarray:
    """Predict proba using best_iteration when early stopping was used."""
    best_n = xgb_best_iteration_plus_one(model)
    if best_n is None:
        return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float32).ravel()

    try:
        return np.asarray(model.predict_proba(X, iteration_range=(0, best_n))[:, 1], dtype=np.float32).ravel()
    except TypeError:
        try:
            return np.asarray(model.predict_proba(X, ntree_limit=best_n)[:, 1], dtype=np.float32).ravel()
        except TypeError:
            return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float32).ravel()


def fit_with_optional_early_stopping(
    est: Any,
    X_tr: Any,
    y_tr: Any,
    *,
    eval_set: Optional[list] = None,
    early_stopping_rounds: int = 0,
    verbose: bool = False,
) -> Any:
    """Fit an XGBoost sklearn estimator with early stopping in a version-compatible way.

    XGBoost 3.x moved early_stopping_rounds from fit() kwargs to estimator params.
    This helper supports both families.

    Notes:
    - We only pass eval_set if fit() supports it.
    - If early stopping can't be configured, we fall back to a normal fit.
    """
    fit_sig = inspect.signature(est.fit)
    fit_kwargs: Dict[str, Any] = {}

    if eval_set is not None and "eval_set" in fit_sig.parameters:
        fit_kwargs["eval_set"] = eval_set
    if "verbose" in fit_sig.parameters:
        fit_kwargs["verbose"] = bool(verbose)

    esr = int(early_stopping_rounds)
    if esr > 0 and eval_set is not None:
        # XGBoost <=2.x: early_stopping_rounds is a fit kwarg
        if "early_stopping_rounds" in fit_sig.parameters:
            fit_kwargs["early_stopping_rounds"] = esr
        else:
            # XGBoost >=3.x: early stopping is controlled via estimator params
            try:
                params = est.get_params()
            except Exception:
                params = {}

            if "early_stopping_rounds" in params:
                try:
                    est.set_params(early_stopping_rounds=esr)
                except Exception:
                    pass
            elif "callbacks" in params:
                # Callback-style fallback (best-effort)
                try:
                    from xgboost.callback import EarlyStopping  # type: ignore
                    est.set_params(callbacks=[EarlyStopping(rounds=esr, save_best=True)])
                except Exception:
                    pass

    return est.fit(X_tr, y_tr, **fit_kwargs)

def inner_group_train_es_split(
    tr_idx: np.ndarray,
    groups: Optional[np.ndarray],
    *,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create an early-stopping split *inside* a training fold.

    - If groups are provided and have >=2 unique values in the fold, uses GroupShuffleSplit.
    - Otherwise uses a simple random split on rows.
    - Returns (train_idx, es_idx) as absolute indices into the full training arrays.
    """
    tr_idx = np.asarray(tr_idx, dtype=int)
    if tr_idx.size == 0:
        return tr_idx, np.asarray([], dtype=int)

    ts = float(test_size)
    ts = min(0.9, max(0.0, ts))
    if ts <= 0.0:
        return tr_idx, np.asarray([], dtype=int)

    rng = np.random.RandomState(int(seed))

    if groups is None:
        perm = rng.permutation(tr_idx)
        n_es = int(max(1, round(ts * perm.size)))
        es = perm[:n_es]
        tr2 = perm[n_es:]
        if tr2.size == 0:
            return tr_idx, np.asarray([], dtype=int)
        return tr2, es

    try:
        import pandas as pd
        from sklearn.model_selection import GroupShuffleSplit
    except Exception:
        # Fallback: row-based split if sklearn isn't available for some reason
        perm = rng.permutation(tr_idx)
        n_es = int(max(1, round(ts * perm.size)))
        es = perm[:n_es]
        tr2 = perm[n_es:]
        if tr2.size == 0:
            return tr_idx, np.asarray([], dtype=int)
        return tr2, es

    g = np.asarray(groups)
    g_fold = g[tr_idx]
    if int(pd.Series(g_fold).nunique()) < 2:
        return tr_idx, np.asarray([], dtype=int)

    splitter = GroupShuffleSplit(n_splits=1, test_size=ts, random_state=int(seed))
    tr_rel, es_rel = next(splitter.split(np.zeros(tr_idx.size), groups=g_fold))
    tr2 = tr_idx[np.asarray(tr_rel, dtype=int)]
    es = tr_idx[np.asarray(es_rel, dtype=int)]
    if tr2.size == 0 or es.size == 0:
        return tr_idx, np.asarray([], dtype=int)
    return tr2, es



def set_n_estimators_if_supported(est: Any, n_estimators: Optional[int]) -> None:
    """Best-effort: set n_estimators if the estimator exposes it."""
    if n_estimators is None:
        return
    try:
        params = est.get_params()
    except Exception:
        return
    if "n_estimators" not in params:
        return
    try:
        est.set_params(n_estimators=int(n_estimators))
    except Exception:
        pass


def _supports_device_param() -> bool:
    """Return True if XGB* estimators support the 'device' param (XGBoost >=2.0)."""
    try:
        from xgboost import XGBRegressor  # type: ignore
        m = XGBRegressor()
        return "device" in m.get_params()
    except Exception:
        return False


def resolve_device(requested: str = "auto", *, allow_fallback: bool = True) -> Tuple[str, Optional[str]]:
    """Resolve XGBoost device string.

    Returns (device, warning_message).

    - requested="auto": uses env XGB_DEVICE if set, else "cpu"
    - requested="cuda"/"cuda:0": tries to verify basic GPU availability; may fall back to cpu
    """
    req = str(requested or "auto").strip().lower()
    if req == "auto":
        req = str(os.environ.get("XGB_DEVICE", "cpu")).strip().lower() or "cpu"

    # Normalize
    if req.startswith("gpu"):
        req = "cuda"
    if req in {"cuda", "cuda:0", "cuda:1"}:
        # Check for nvidia-smi as a quick proxy. If missing, fall back.
        try:
            smi = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if smi.returncode != 0:
                raise RuntimeError(smi.stderr.strip() or smi.stdout.strip())
        except Exception as e:
            if allow_fallback:
                return "cpu", f"GPU device requested ({requested!r}) but NVIDIA tooling/GPU not available: {e}. Falling back to CPU."
            return req, None

    return req, None


def make_xgb_common_params(
    *,
    device: str = "cpu",
    n_jobs: int = 4,
    seed: int = 42,
    max_bin: int = 256,
    tree_method: str = "hist",
    verbosity: int = 0,
) -> Dict[str, Any]:
    """Create common XGBoost params in a compatibility-friendly way."""
    params: Dict[str, Any] = dict(
        n_jobs=int(n_jobs),
        seed=int(seed),
        random_state=int(seed),
        verbosity=int(verbosity),
        max_bin=int(max_bin),
    )

    if _supports_device_param():
        # XGBoost >=2.0
        params["tree_method"] = str(tree_method)
        params["device"] = str(device)
        return params

    # XGBoost <2.0 fallback
    # - CPU: tree_method='hist'
    # - GPU: tree_method='gpu_hist' + predictor='gpu_predictor'
    if str(device).lower().startswith("cuda"):
        params["tree_method"] = "gpu_hist"
        params["predictor"] = "gpu_predictor"
        # gpu_id optional; not always needed
        try:
            gpu_id = int(str(device).split(":")[1])
        except Exception:
            gpu_id = 0
        params["gpu_id"] = gpu_id
    else:
        params["tree_method"] = str(tree_method)
    return params