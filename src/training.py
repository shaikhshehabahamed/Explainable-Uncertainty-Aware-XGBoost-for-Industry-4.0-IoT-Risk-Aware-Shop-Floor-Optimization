# src/training.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import json
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import GroupKFold

from .artifacts import ClassificationArtifact, ModelBundle, RegressionArtifact
from .data import (
    CLASSIFICATION_TARGETS,
    CONTEXT_FEATURES,
    DECISION_LEVERS,
    PROB_REGRESSION_TARGETS,
    REGRESSION_TARGETS,
)
from .metrics import (
    classification_metrics,
    compute_scale_pos_weight,
    regression_metrics,
    select_threshold,
    validate_binary_labels,
)
from .models import (
    DEFAULT_CLF_PARAMS,
    DEFAULT_DEVICE,
    DEFAULT_MAX_BIN,
    DEFAULT_N_JOBS,
    DEFAULT_REG_PARAMS,
    DEFAULT_SEED,
    DEFAULT_TREE_METHOD,
    build_pipeline_for_estimator,
    make_classifier,
    make_regressor,
)
from .transforms import TargetTransform
from .uncertainty import conformal_quantile, fit_probability_bounds_from_calibration
from .xgb_utils import (
    fit_with_optional_early_stopping,
    inner_group_train_es_split,
    resolve_device,
    set_n_estimators_if_supported,
    xgb_best_iteration_plus_one,
    xgb_predict_best,
    xgb_predict_proba_best,
)


@dataclass(frozen=True)
class TrainingConfig:
    device: str = "auto"
    seed: int = DEFAULT_SEED
    n_jobs: int = DEFAULT_N_JOBS
    max_bin: int = DEFAULT_MAX_BIN
    tree_method: str = DEFAULT_TREE_METHOD

    n_splits: int = 5

    # Early stopping inside folds (nested split inside training fold)
    early_stopping_rounds: int = 100
    early_stopping_test_size: float = 0.2

    # Uncertainty calibration
    pi_alpha: float = 0.1  # 90% PI
    clf_bounds_alpha: float = 0.1
    clf_bounds_bins: int = 10

    # Classification decision rule
    threshold_metric: str = "f1"

    # Hyperparameter optimization (random search)
    hpo_trials: int = 0
    hpo_seed: int = 123
    hpo_max_rows: int = 2500  # subsample rows during HPO for speed


def _ensure_outdirs(outdir: str) -> Dict[str, Path]:
    base = Path(outdir)
    paths = {
        "base": base,
        "models": base / "models",
        "metrics": base / "metrics",
        "predictions": base / "predictions",
        "hpo": base / "hpo",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _groupkfold_splits(X: pd.DataFrame, groups: pd.Series, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return deterministic GroupKFold splits, with clear errors on invalid group counts.

    Notes:
    - GroupKFold requires at least 2 distinct groups and n_splits <= n_groups.
    - GroupKFold itself is deterministic given the ordering of groups; `seed` is stored for metadata only.
    """
    g = pd.Series(groups)
    n_groups = int(g.nunique())

    if n_groups < 2:
        raise ValueError(
            f"GroupKFold requires at least 2 distinct groups, but only {n_groups} group was found in the training data."
        )

    n_splits_eff = int(min(max(2, int(n_splits)), n_groups))
    gkf = GroupKFold(n_splits=n_splits_eff)

    splits = [(np.asarray(tr, dtype=int), np.asarray(va, dtype=int)) for tr, va in gkf.split(X, groups=groups)]
    return splits

def _sample_hpo_params_reg(rng: np.random.RandomState) -> Dict[str, Any]:
    return {
        "learning_rate": float(rng.uniform(0.01, 0.08)),
        "max_depth": int(rng.choice([3, 4, 5, 6, 7, 8])),
        "min_child_weight": float(rng.uniform(1.0, 10.0)),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "reg_lambda": float(rng.uniform(0.5, 6.0)),
        "reg_alpha": float(rng.uniform(0.0, 2.0)),
    }


def _sample_hpo_params_clf(rng: np.random.RandomState) -> Dict[str, Any]:
    p = _sample_hpo_params_reg(rng)
    p["max_delta_step"] = float(rng.uniform(0.0, 5.0))
    return p


def _evaluate_params_reg_oof(
    X: pd.DataFrame,
    y_t: np.ndarray,
    groups: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    config: TrainingConfig,
    params: Dict[str, Any],
) -> float:
    """Return mean RMSE across folds (lower is better). y_t is already transformed if needed."""
    # Subsample rows for speed if configured
    if config.hpo_max_rows and X.shape[0] > config.hpo_max_rows:
        rng = np.random.RandomState(config.hpo_seed)
        idx = rng.choice(np.arange(X.shape[0]), size=int(config.hpo_max_rows), replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y_t = y_t[idx]
        groups = groups[idx]
        splits = _groupkfold_splits(X, pd.Series(groups), config.n_splits, config.seed)

    rmse_list: List[float] = []

    # We fit preprocessing inside each fold to avoid leakage.
    # Using a Pipeline directly for early stopping is tricky across XGB versions, so we fit XGB on transformed matrices.
    from .data import build_preprocessor, get_numeric_and_categorical_features
    from .models import ToFloat32

    num_cols, cat_cols = get_numeric_and_categorical_features()
    caster = ToFloat32()

    for fold_i, (tr_idx, va_idx) in enumerate(splits, start=1):
        tr_idx = np.asarray(tr_idx, dtype=int)
        va_idx = np.asarray(va_idx, dtype=int)

        tr2_idx, es_idx = inner_group_train_es_split(
            tr_idx, groups, test_size=float(config.early_stopping_test_size), seed=int(config.seed) + int(fold_i)
        )

        X_tr2 = X.iloc[tr2_idx]
        y_tr2 = y_t[tr2_idx]

        X_va = X.iloc[va_idx]
        y_va = y_t[va_idx]

        pre = build_preprocessor(num_cols, cat_cols)
        pre.fit(X_tr2)

        Xt_tr2 = caster.transform(pre.transform(X_tr2))
        Xt_va = caster.transform(pre.transform(X_va))

        est = make_regressor(
            device=config.device,
            seed=config.seed,
            n_jobs=config.n_jobs,
            max_bin=config.max_bin,
            tree_method=config.tree_method,
            params=params,
        )

        if int(config.early_stopping_rounds) > 0 and es_idx.size > 0:
            X_es = X.iloc[es_idx]
            y_es = y_t[es_idx]
            Xt_es = caster.transform(pre.transform(X_es))
            fit_with_optional_early_stopping(
                est,
                Xt_tr2,
                y_tr2,
                eval_set=[(Xt_es, y_es)],
                early_stopping_rounds=int(config.early_stopping_rounds),
                verbose=False,
            )
        else:
            est.fit(Xt_tr2, y_tr2)

        pred = xgb_predict_best(est, Xt_va)
        # RMSE on transformed scale for HPO consistency; it's monotone wrt original in identity.
        rmse = float(np.sqrt(np.mean((y_va - pred) ** 2)))
        rmse_list.append(rmse)

    return float(np.mean(rmse_list)) if rmse_list else float("inf")


def _evaluate_params_clf_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    config: TrainingConfig,
    params: Dict[str, Any],
    scale_pos_weight: float,
) -> float:
    """Return mean logloss across folds (lower is better)."""
    if config.hpo_max_rows and X.shape[0] > config.hpo_max_rows:
        rng = np.random.RandomState(config.hpo_seed)
        idx = rng.choice(np.arange(X.shape[0]), size=int(config.hpo_max_rows), replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y[idx]
        groups = groups[idx]
        splits = _groupkfold_splits(X, pd.Series(groups), config.n_splits, config.seed)

    ll_list: List[float] = []

    from .data import build_preprocessor, get_numeric_and_categorical_features
    from .models import ToFloat32

    num_cols, cat_cols = get_numeric_and_categorical_features()
    caster = ToFloat32()

    for fold_i, (tr_idx, va_idx) in enumerate(splits, start=1):
        tr_idx = np.asarray(tr_idx, dtype=int)
        va_idx = np.asarray(va_idx, dtype=int)

        y_tr = y[tr_idx]
        if len(np.unique(y_tr)) < 2:
            # Degenerate fold => skip (cannot compute logloss reliably)
            continue

        tr2_idx, es_idx = inner_group_train_es_split(
            tr_idx, groups, test_size=float(config.early_stopping_test_size), seed=int(config.seed) + int(fold_i)
        )

        X_tr2 = X.iloc[tr2_idx]
        y_tr2 = y[tr2_idx]
        X_va = X.iloc[va_idx]
        y_va = y[va_idx]

        pre = build_preprocessor(num_cols, cat_cols)
        pre.fit(X_tr2)

        Xt_tr2 = caster.transform(pre.transform(X_tr2))
        Xt_va = caster.transform(pre.transform(X_va))

        est = make_classifier(
            device=config.device,
            seed=config.seed,
            n_jobs=config.n_jobs,
            max_bin=config.max_bin,
            tree_method=config.tree_method,
            scale_pos_weight=float(scale_pos_weight),
            params=params,
        )

        if int(config.early_stopping_rounds) > 0 and es_idx.size > 0:
            X_es = X.iloc[es_idx]
            y_es = y[es_idx]
            Xt_es = caster.transform(pre.transform(X_es))
            fit_with_optional_early_stopping(
                est,
                Xt_tr2,
                y_tr2,
                eval_set=[(Xt_es, y_es)],
                early_stopping_rounds=int(config.early_stopping_rounds),
                verbose=False,
            )
        else:
            est.fit(Xt_tr2, y_tr2)

        proba = xgb_predict_proba_best(est, Xt_va).astype(float)

        # logloss
        eps = 1e-9
        proba = np.clip(proba, eps, 1.0 - eps)
        ll = -float(np.mean(y_va * np.log(proba) + (1 - y_va) * np.log(1 - proba)))
        ll_list.append(ll)

    return float(np.mean(ll_list)) if ll_list else float("inf")


def _random_search_best_params(
    task: str,
    X_train: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    config: TrainingConfig,
    scale_pos_weight: Optional[float] = None,
    transform: Optional[TargetTransform] = None,
) -> Dict[str, Any]:
    """Return best param overrides (random search)."""
    n_trials = int(max(0, config.hpo_trials))
    if n_trials <= 0:
        return {}

    rng = np.random.RandomState(int(config.hpo_seed))
    best_params: Dict[str, Any] = {}
    best_score = float("inf")

    # Transform only applies to regression HPO if requested
    y_for_hpo = y
    if task == "regression" and transform is not None:
        y_for_hpo = transform.forward(y_for_hpo)

    for t in range(1, n_trials + 1):
        cand = _sample_hpo_params_reg(rng) if task == "regression" else _sample_hpo_params_clf(rng)

        if task == "regression":
            score = _evaluate_params_reg_oof(X_train, y_for_hpo, groups, splits, config=config, params=cand)
        else:
            if scale_pos_weight is None:
                raise ValueError("scale_pos_weight is required for classification HPO.")
            score = _evaluate_params_clf_oof(
                X_train, y.astype(int), groups, splits, config=config, params=cand, scale_pos_weight=scale_pos_weight
            )

        if score < best_score:
            best_score = float(score)
            best_params = dict(cand)

    return best_params


def _lever_bounds_from_training(X_train: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Compute robust bounds for each decision lever from training data (1st..99th pct)."""
    bounds: Dict[str, Dict[str, Any]] = {}
    for col in DECISION_LEVERS:
        s = X_train[col]
        is_int = str(s.dtype).startswith(("int", "uint"))
        lo = float(np.nanpercentile(s.to_numpy(dtype=float), 1))
        hi = float(np.nanpercentile(s.to_numpy(dtype=float), 99))
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
        if is_int:
            lo = float(np.floor(lo))
            hi = float(np.ceil(hi))
        bounds[col] = {"low": lo, "high": hi, "dtype": "int" if is_int else "float"}
    return bounds


def train_all_targets(
    X_train_raw: pd.DataFrame,
    y_train_all: pd.DataFrame,
    groups_train: pd.Series,
    *,
    outdir: str = "outputs",
    config: Optional[TrainingConfig] = None,
) -> Tuple[ModelBundle, Dict[str, Any]]:
    """Train + calibrate uncertainty for all targets, save artifacts, and return a bundle + metrics.

    Design goals:
    - Group-aware CV (GroupKFold) to avoid leakage across Unit_ID.
    - Uncertainty calibration using strictly out-of-fold (OOF) predictions.
    - Avoid repeated work: within each CV fold we fit/transform preprocessing once and re-use for all targets.
    """
    cfg0 = config or TrainingConfig()

    # Resolve device (CPU fallback is allowed)
    device, warn = resolve_device(cfg0.device, allow_fallback=True)
    device_warning = warn

    cfg = TrainingConfig(
        device=device,
        seed=cfg0.seed,
        n_jobs=cfg0.n_jobs,
        max_bin=cfg0.max_bin,
        tree_method=cfg0.tree_method,
        n_splits=cfg0.n_splits,
        early_stopping_rounds=cfg0.early_stopping_rounds,
        early_stopping_test_size=cfg0.early_stopping_test_size,
        pi_alpha=cfg0.pi_alpha,
        clf_bounds_alpha=cfg0.clf_bounds_alpha,
        clf_bounds_bins=cfg0.clf_bounds_bins,
        threshold_metric=cfg0.threshold_metric,
        hpo_trials=cfg0.hpo_trials,
        hpo_seed=cfg0.hpo_seed,
        hpo_max_rows=cfg0.hpo_max_rows,
    )

    out_paths = _ensure_outdirs(outdir)

    groups_arr = groups_train.to_numpy()
    splits = _groupkfold_splits(X_train_raw, groups_train, cfg.n_splits, cfg.seed)

    # Cache preprocessing components
    from .data import build_preprocessor, get_numeric_and_categorical_features
    from .models import ToFloat32

    num_cols, cat_cols = get_numeric_and_categorical_features()
    caster = ToFloat32()

    # -----------------------------
    # Prepare targets + optional HPO
    # -----------------------------
    reg_transform: Dict[str, TargetTransform] = {}
    y_reg_orig: Dict[str, np.ndarray] = {}
    y_reg_t: Dict[str, np.ndarray] = {}
    oof_pred_t: Dict[str, np.ndarray] = {}
    best_ns_reg: Dict[str, List[int]] = {}
    reg_overrides: Dict[str, Dict[str, Any]] = {}

    for t in REGRESSION_TARGETS:
        y = y_train_all[t].to_numpy(dtype=float)
        if pd.isna(y).any():
            raise ValueError(f"Regression target '{t}' contains NaN labels; please clean/drop or impute.")
        trf = TargetTransform("logit") if t in PROB_REGRESSION_TARGETS else TargetTransform("identity")
        reg_transform[t] = trf
        y_reg_orig[t] = y
        y_reg_t[t] = trf.forward(y)
        oof_pred_t[t] = np.empty((y.shape[0],), dtype=np.float32)
        best_ns_reg[t] = []

        reg_overrides[t] = _random_search_best_params(
            "regression", X_train_raw, y, groups_arr, splits, config=cfg, transform=trf
        )

    # Classification targets: labels + scale_pos_weight
    y_clf: Dict[str, np.ndarray] = {}
    spw: Dict[str, float] = {}
    base_rate: Dict[str, float] = {}
    p_oof: Dict[str, np.ndarray] = {}
    best_ns_clf: Dict[str, List[int]] = {}
    clf_overrides: Dict[str, Dict[str, Any]] = {}

    for t in CLASSIFICATION_TARGETS:
        yi = validate_binary_labels(y_train_all[t].to_numpy(), t)
        y_clf[t] = yi
        spw[t] = compute_scale_pos_weight(yi)
        base_rate[t] = float(np.clip(np.mean(yi), 1e-6, 1.0 - 1e-6))
        p_oof[t] = np.empty((yi.shape[0],), dtype=np.float32)
        best_ns_clf[t] = []
        clf_overrides[t] = _random_search_best_params(
            "classification", X_train_raw, yi, groups_arr, splits, config=cfg, scale_pos_weight=spw[t]
        )

    # -----------------------------
    # OoF training loops (fold-first to avoid repeated preprocessing)
    # -----------------------------
    fold_rows: List[Dict[str, Any]] = []

    for fold_i, (tr_idx, va_idx) in enumerate(splits, start=1):
        tr_idx = np.asarray(tr_idx, dtype=int)
        va_idx = np.asarray(va_idx, dtype=int)

        # Inner split for early stopping (group-aware) — same indices for all targets in this fold
        tr2_abs, es_abs = inner_group_train_es_split(
            tr_idx,
            groups_arr,
            test_size=float(cfg.early_stopping_test_size),
            seed=int(cfg.seed) + int(fold_i),
        )

        X_tr = X_train_raw.iloc[tr_idx]
        X_va = X_train_raw.iloc[va_idx]
        X_tr2 = X_train_raw.iloc[tr2_abs] if tr2_abs.size > 0 else X_tr

        # Fit preprocessing only on the data used to fit the estimators (tr2) to avoid leakage into the
        # early-stopping set.
        pre = build_preprocessor(num_cols, cat_cols)
        pre.fit(X_tr2)

        Xt_tr_all = caster.transform(pre.transform(X_tr))
        Xt_va = caster.transform(pre.transform(X_va))
        Xt_tr2 = caster.transform(pre.transform(X_tr2))

        Xt_es = None
        if es_abs.size > 0:
            X_es = X_train_raw.iloc[es_abs]
            Xt_es = caster.transform(pre.transform(X_es))

        # -------------------------
        # Regression targets
        # -------------------------
        for t in REGRESSION_TARGETS:
            est = make_regressor(
                device=cfg.device,
                seed=cfg.seed,
                n_jobs=cfg.n_jobs,
                max_bin=cfg.max_bin,
                tree_method=cfg.tree_method,
                params=reg_overrides.get(t, {}),
            )

            y_tr2 = y_reg_t[t][tr2_abs]
            if int(cfg.early_stopping_rounds) > 0 and Xt_es is not None and es_abs.size > 0:
                y_es = y_reg_t[t][es_abs]
                fit_with_optional_early_stopping(
                    est,
                    Xt_tr2,
                    y_tr2,
                    eval_set=[(Xt_es, y_es)],
                    early_stopping_rounds=int(cfg.early_stopping_rounds),
                    verbose=False,
                )
            else:
                est.fit(Xt_tr2, y_tr2)

            best_n = xgb_best_iteration_plus_one(est)
            if best_n is not None and best_n > 0:
                best_ns_reg[t].append(int(best_n))

            pred_t = xgb_predict_best(est, Xt_va)
            oof_pred_t[t][va_idx] = pred_t

            # Metrics on original scale
            y_va = y_reg_orig[t][va_idx]
            pred = reg_transform[t].inverse(pred_t)
            m = regression_metrics(y_va, pred)
            fold_rows.append({"target": t, "task_type": "regression", "fold": int(fold_i), **m})

        # -------------------------
        # Classification targets
        # -------------------------
        for t in CLASSIFICATION_TARGETS:
            y_all = y_clf[t]
            y_tr_fold = y_all[tr_idx]

            if len(np.unique(y_tr_fold)) < 2:
                # Degenerate fold => fall back to the global base rate (avoids extreme 0/1 probs)
                p_oof[t][va_idx] = float(base_rate[t])
                continue

            # If inner split causes training to become degenerate, disable early stopping for this target+fold.
            use_es = int(cfg.early_stopping_rounds) > 0 and Xt_es is not None and es_abs.size > 0
            y_tr2 = y_all[tr2_abs]
            if (len(np.unique(y_tr2)) < 2) or (not use_es):
                Xt_train = Xt_tr_all
                y_train = y_tr_fold
                eval_set = None
                es_rounds = 0
            else:
                Xt_train = Xt_tr2
                y_train = y_tr2
                y_es = y_all[es_abs]
                eval_set = [(Xt_es, y_es)]
                es_rounds = int(cfg.early_stopping_rounds)

            est = make_classifier(
                device=cfg.device,
                seed=cfg.seed,
                n_jobs=cfg.n_jobs,
                max_bin=cfg.max_bin,
                tree_method=cfg.tree_method,
                scale_pos_weight=float(spw[t]),
                params=clf_overrides.get(t, {}),
            )

            if es_rounds > 0 and eval_set is not None:
                fit_with_optional_early_stopping(
                    est,
                    Xt_train,
                    y_train,
                    eval_set=eval_set,
                    early_stopping_rounds=es_rounds,
                    verbose=False,
                )
            else:
                est.fit(Xt_train, y_train)

            best_n = xgb_best_iteration_plus_one(est)
            if best_n is not None and best_n > 0:
                best_ns_clf[t].append(int(best_n))

            p_oof[t][va_idx] = xgb_predict_proba_best(est, Xt_va)

    # -----------------------------
    # Build final artifacts + calibration
    # -----------------------------
    reg_artifacts: Dict[str, RegressionArtifact] = {}
    clf_artifacts: Dict[str, ClassificationArtifact] = {}
    metrics_by_target: Dict[str, Any] = {}

    # Regression: conformal q + final fit
    for t in REGRESSION_TARGETS:
        abs_resid = np.abs(y_reg_t[t] - oof_pred_t[t])
        q = conformal_quantile(abs_resid, alpha=float(cfg.pi_alpha))
        n_estimators_final: Optional[int] = int(np.nanmedian(best_ns_reg[t])) if best_ns_reg[t] else None

        final_est = make_regressor(
            device=cfg.device,
            seed=cfg.seed,
            n_jobs=cfg.n_jobs,
            max_bin=cfg.max_bin,
            tree_method=cfg.tree_method,
            params=reg_overrides.get(t, {}),
        )
        set_n_estimators_if_supported(final_est, n_estimators_final)

        final_pipe = build_pipeline_for_estimator(final_est)
        final_pipe.fit(X_train_raw, y_reg_t[t])

        art = RegressionArtifact(
            target=t,
            pipeline=final_pipe,
            abs_resid_q=float(q),
            alpha=float(cfg.pi_alpha),
            transform=reg_transform[t],
        )
        reg_artifacts[t] = art
        dump(art, str(out_paths["models"] / f"{t}.joblib"))

        df_f = pd.DataFrame([r for r in fold_rows if r["target"] == t and r["task_type"] == "regression"])
        metrics_by_target[t] = {
            "task_type": "regression",
            "oof_mean": {k: float(np.nanmean(df_f[k])) for k in ["rmse", "mae"] if k in df_f},
            "conformal_q": float(q),
            "transform": reg_transform[t].name,
            "hpo_best_overrides": reg_overrides.get(t, {}),
            "n_estimators_final": int(n_estimators_final) if n_estimators_final else None,
        }

    # Classification: cross-fitted isotonic calibration + bounds + threshold + final fit
    for t in CLASSIFICATION_TARGETS:
        y = y_clf[t]
        p = np.clip(p_oof[t], 0.0, 1.0)

        from sklearn.isotonic import IsotonicRegression

        # ---------------------------------------------------------
        # Cross-fitted isotonic calibration (prevents calibrator bias)
        # ---------------------------------------------------------
        p_cal_oof = np.empty_like(p, dtype=np.float32)

        for fold_i, (tr_idx, va_idx) in enumerate(splits, start=1):
            tr_idx = np.asarray(tr_idx, dtype=int)
            va_idx = np.asarray(va_idx, dtype=int)

            y_tr = y[tr_idx]
            p_tr = p[tr_idx]

            # If the fold training labels are degenerate, isotonic is undefined.
            # In that case, fall back to identity calibration for this fold.
            if len(np.unique(y_tr)) < 2:
                p_cal_oof[va_idx] = np.clip(p[va_idx], 0.0, 1.0).astype(np.float32)
                continue

            iso_fold = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso_fold.fit(p_tr, y_tr)
            p_cal_oof[va_idx] = np.asarray(iso_fold.predict(p[va_idx]), dtype=np.float32).ravel()

        # Fit a final calibrator on all OOF predictions for deployment/inference
        iso: Optional[IsotonicRegression] = None
        if len(np.unique(y)) >= 2:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(p, y)

        # Bounds are fit in calibrated-probability space as used at inference time.
        p_cal_all = (
            np.asarray(iso.predict(p), dtype=np.float32).ravel()
            if iso is not None
            else np.clip(p, 0.0, 1.0).astype(np.float32)
        )

        bounds = fit_probability_bounds_from_calibration(
            p_cal_all,
            y,
            alpha=float(cfg.clf_bounds_alpha),
            n_bins=int(cfg.clf_bounds_bins),
        )

        # Threshold selection uses cross-fitted calibrated probabilities
        thr = (
            float(select_threshold(y, p_cal_oof, metric=str(cfg.threshold_metric)))
            if len(np.unique(y)) >= 2
            else 0.5
        )

        # Fold metrics using cross-fitted calibrated OOF probabilities
        for fold_i, (_, va_idx) in enumerate(splits, start=1):
            va_idx = np.asarray(va_idx, dtype=int)
            m = classification_metrics(y[va_idx], p_cal_oof[va_idx], threshold=float(thr))
            fold_rows.append({"target": t, "task_type": "classification", "fold": int(fold_i), **m})

        n_estimators_final: Optional[int] = int(np.nanmedian(best_ns_clf[t])) if best_ns_clf[t] else None

        final_est = make_classifier(
            device=cfg.device,
            seed=cfg.seed,
            n_jobs=cfg.n_jobs,
            max_bin=cfg.max_bin,
            tree_method=cfg.tree_method,
            scale_pos_weight=float(spw[t]),
            params=clf_overrides.get(t, {}),
        )
        set_n_estimators_if_supported(final_est, n_estimators_final)

        final_pipe = build_pipeline_for_estimator(final_est)
        final_pipe.fit(X_train_raw, y)

        art = ClassificationArtifact(
            target=t,
            pipeline=final_pipe,
            isotonic=iso,
            bounds=bounds,
            decision_threshold=float(thr),
        )
        clf_artifacts[t] = art
        dump(art, str(out_paths["models"] / f"{t}.joblib"))

        df_f = pd.DataFrame([r for r in fold_rows if r["target"] == t and r["task_type"] == "classification"])
        mean_metrics = {k: float(np.nanmean(df_f[k])) for k in df_f.columns if k not in {"target", "task_type", "fold"}}

        metrics_by_target[t] = {
            "task_type": "classification",
            "oof_mean": mean_metrics,
            "decision_threshold": float(thr),
            "bounds_alpha": float(cfg.clf_bounds_alpha),
            "bounds_bins": int(cfg.clf_bounds_bins),
            "hpo_best_overrides": clf_overrides.get(t, {}),
            "n_estimators_final": int(n_estimators_final) if n_estimators_final else None,
        }

    # -----------------------------
    # Bundle + persistence
    # -----------------------------
    lever_bounds = _lever_bounds_from_training(X_train_raw)
    bundle = ModelBundle(
        regression=reg_artifacts,
        classification=clf_artifacts,
        metadata={
            "device": cfg.device,
            "device_warning": device_warning,
            "seed": int(cfg.seed),
            "n_jobs": int(cfg.n_jobs),
            "max_bin": int(cfg.max_bin),
            "tree_method": str(cfg.tree_method),
            "n_splits": int(cfg.n_splits),
            "early_stopping_rounds": int(cfg.early_stopping_rounds),
            "early_stopping_test_size": float(cfg.early_stopping_test_size),
            "pi_alpha": float(cfg.pi_alpha),
            "clf_bounds_alpha": float(cfg.clf_bounds_alpha),
            "clf_bounds_bins": int(cfg.clf_bounds_bins),
            "threshold_metric": str(cfg.threshold_metric),
            "targets": {
                "regression": list(reg_artifacts.keys()),
                "classification": list(clf_artifacts.keys()),
            },
            "features": {
                "context": list(CONTEXT_FEATURES),
                "decision_levers": list(DECISION_LEVERS),
            },
            "lever_bounds": lever_bounds,
        },
    )
    bundle_path = bundle.save(str(out_paths["models"] / "bundle.joblib"))

    # Save metrics CSV (fold rows)
    df_folds = pd.DataFrame(fold_rows)
    df_folds.to_csv(out_paths["metrics"] / "train_oof_folds.csv", index=False)

    # Save summary JSON
    summary = {
        "bundle_path": bundle_path,
        "by_target": metrics_by_target,
    }
    with open(out_paths["metrics"] / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return bundle, summary

def evaluate_on_test(
    bundle: ModelBundle,
    X_test_raw: pd.DataFrame,
    y_test_all: pd.DataFrame,
    *,
    outdir: str = "outputs",
) -> Dict[str, Any]:
    """Evaluate bundle artifacts on a held-out test set (no refitting)."""
    out_paths = _ensure_outdirs(outdir)

    rows: List[Dict[str, Any]] = []
    by_target: Dict[str, Any] = {}

    # Regression
    for target, art in bundle.regression.items():
        y_true = y_test_all[target].to_numpy(dtype=float)
        mean, lo, hi = art.predict_mean_and_bounds(X_test_raw)
        m = regression_metrics(y_true, mean)
        coverage = float(np.mean((y_true >= lo) & (y_true <= hi)))
        row = {"target": target, "task_type": "regression", **m, "pi_coverage": coverage}
        rows.append(row)
        by_target[target] = row

    # Classification
    for target, art in bundle.classification.items():
        y_true = validate_binary_labels(y_test_all[target].to_numpy(), target)
        p_cal, p_lo, p_hi = art.predict_bounds(X_test_raw)
        m = classification_metrics(y_true, p_cal, threshold=float(art.decision_threshold))
        # Additional uncertainty diagnostics
        row = {
            "target": target,
            "task_type": "classification",
            **m,
            "mean_p_cal": float(np.mean(p_cal)),
            "mean_p_lower": float(np.mean(p_lo)),
            "mean_p_upper": float(np.mean(p_hi)),
        }
        rows.append(row)
        by_target[target] = row

    df = pd.DataFrame(rows)
    df.to_csv(out_paths["metrics"] / "test_metrics.csv", index=False)

    summary = {"by_target": by_target, "rows": rows}
    with open(out_paths["metrics"] / "test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary