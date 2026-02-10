# src/explain.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .artifacts import ModelBundle
from .data import CONTEXT_FEATURES, DECISION_LEVERS


RANDOM_STATE = 42


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sample_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0:
        raise ValueError("n must be positive.")
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, replace=False, random_state=int(seed)).copy()


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _logloss(y_true: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int).ravel()
    p = np.asarray(p, dtype=float).ravel()
    eps = 1e-9
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def permutation_importance(
    bundle: ModelBundle,
    X: pd.DataFrame,
    y: pd.DataFrame,
    *,
    outdir: str = "outputs",
    sample_n: int = 1200,
    n_repeats: int = 3,
    seed: int = RANDOM_STATE,
) -> str:
    """
    Dependency-light global explainability:
      - Permutation importance for every target (all outcomes)
      - Saves CSV: outputs/explain/permutation_importance.csv
      - Saves one plot per target: outputs/explain/perm_importance_<target>.png
    """
    out_base = _ensure_dir(Path(outdir) / "explain")
    Xs = _sample_df(X, n=sample_n, seed=seed)  # keep original indices
    ys = y.loc[Xs.index] if len(y) == len(X) else y.copy()
    Xs = Xs.reset_index(drop=True)
    ys = ys.reset_index(drop=True)

    # Features to permute
    features = CONTEXT_FEATURES + DECISION_LEVERS

    rng = np.random.RandomState(int(seed))
    rows: List[Dict[str, Any]] = []

    def _permute_col(X_in: pd.DataFrame, col: str) -> pd.DataFrame:
        Xp = X_in.copy()
        Xp[col] = rng.permutation(Xp[col].to_numpy())
        return Xp

    # Regression targets
    for t, art in bundle.regression.items():
        y_true = ys[t].to_numpy(dtype=float)
        pred_mean, _, _ = art.predict_mean_and_bounds(Xs)
        base = _rmse(y_true, pred_mean)

        for col in features:
            deltas: List[float] = []
            for _ in range(int(n_repeats)):
                Xp = _permute_col(Xs, col)
                p_mean, _, _ = art.predict_mean_and_bounds(Xp)
                score = _rmse(y_true, p_mean)
                deltas.append(score - base)
            imp = float(np.mean(deltas))
            rows.append({"target": t, "task_type": "regression", "feature": col, "importance": imp})

        # Plot top 20
        df_t = pd.DataFrame([r for r in rows if r["target"] == t]).sort_values("importance", ascending=False).head(20)
        try:
            plt.figure(figsize=(10, 6))
            plt.barh(df_t["feature"][::-1], df_t["importance"][::-1])
            plt.xlabel("Permutation importance (ΔRMSE)")
            plt.title(f"Permutation importance — {t}")
            plt.tight_layout()
            plt.savefig(out_base / f"perm_importance_{t}.png", dpi=150)
            plt.close()
        except Exception:
            pass

    # Classification targets
    for t, art in bundle.classification.items():
        y_true = ys[t].to_numpy(dtype=int)
        p_cal, _, _ = art.predict_bounds(Xs)
        base = _logloss(y_true, p_cal)

        for col in features:
            deltas = []
            for _ in range(int(n_repeats)):
                Xp = _permute_col(Xs, col)
                p_cal_p, _, _ = art.predict_bounds(Xp)
                score = _logloss(y_true, p_cal_p)
                deltas.append(score - base)
            imp = float(np.mean(deltas))
            rows.append({"target": t, "task_type": "classification", "feature": col, "importance": imp})

        df_t = pd.DataFrame([r for r in rows if r["target"] == t]).sort_values("importance", ascending=False).head(20)
        try:
            plt.figure(figsize=(10, 6))
            plt.barh(df_t["feature"][::-1], df_t["importance"][::-1])
            plt.xlabel("Permutation importance (ΔLogLoss)")
            plt.title(f"Permutation importance — {t}")
            plt.tight_layout()
            plt.savefig(out_base / f"perm_importance_{t}.png", dpi=150)
            plt.close()
        except Exception:
            pass

    df_all = pd.DataFrame(rows)
    out_csv = out_base / "permutation_importance.csv"
    df_all.to_csv(out_csv, index=False)
    return str(out_csv)


def what_if_lever_sensitivity(
    bundle: ModelBundle,
    *,
    scenario_row: pd.Series,
    outdir: str = "outputs",
    n_points: int = 15,
    seed: int = RANDOM_STATE,
) -> str:
    """
    Local explainability without SHAP:
      - For a single scenario row, vary each decision lever across its training bounds
      - Predict all targets and save curves
      - Output: outputs/explain/whatif_curves.csv
    """
    out_base = _ensure_dir(Path(outdir) / "explain")
    bounds = bundle.metadata.get("lever_bounds", {})
    if not bounds:
        raise ValueError("Bundle has no lever_bounds metadata; cannot run what-if analysis.")

    ctx = scenario_row[CONTEXT_FEATURES].to_frame().T

    rows: List[Dict[str, Any]] = []
    for lv in DECISION_LEVERS:
        b = bounds.get(lv)
        if not b:
            continue
        lo = float(b["low"])
        hi = float(b["high"])
        dtype = str(b.get("dtype", "float"))

        # Generate lever values (quantiles in range)
        vals = np.linspace(lo, hi, int(max(2, n_points)))
        if dtype == "int":
            vals = np.unique(np.round(vals).astype(int)).astype(float)

        for v in vals:
            X_row = ctx.copy()
            # Keep other levers at their scenario value if present, else midpoint
            for lv2 in DECISION_LEVERS:
                if lv2 == lv:
                    X_row[lv2] = v
                else:
                    if lv2 in scenario_row.index:
                        X_row[lv2] = scenario_row[lv2]
                    else:
                        bb = bounds.get(lv2, {"low": 0.0, "high": 1.0})
                        X_row[lv2] = (float(bb["low"]) + float(bb["high"])) / 2.0

            rec: Dict[str, Any] = {"lever": lv, "lever_value": float(v)}
            # Regression
            for t, art in bundle.regression.items():
                mean, lo_t, hi_t = art.predict_mean_and_bounds(X_row)
                rec[f"{t}_mean"] = float(mean[0])
                rec[f"{t}_lower90"] = float(lo_t[0])
                rec[f"{t}_upper90"] = float(hi_t[0])
            # Classification
            for t, art in bundle.classification.items():
                p, lb, ub = art.predict_bounds(X_row)
                rec[f"{t}_p_cal"] = float(p[0])
                rec[f"{t}_p_lower"] = float(lb[0])
                rec[f"{t}_p_upper"] = float(ub[0])
                rec[f"{t}_label"] = int(art.predict_label(X_row)[0])

            # Add context for traceability
            for c in CONTEXT_FEATURES:
                rec[f"ctx_{c}"] = scenario_row[c]
            rows.append(rec)

    out_csv = out_base / "whatif_curves.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Optional: quick plots for key outcomes
    key_targets = ["Energy_Used", "Completion_Time", "Predictive_Failure_Score"]
    try:
        df = pd.DataFrame(rows)
        for lv in DECISION_LEVERS:
            dfl = df[df["lever"] == lv].sort_values("lever_value")
            if dfl.empty:
                continue
            for t in key_targets:
                if f"{t}_mean" not in dfl.columns:
                    continue
                plt.figure()
                plt.plot(dfl["lever_value"], dfl[f"{t}_mean"], marker="o")
                plt.fill_between(dfl["lever_value"], dfl[f"{t}_lower90"], dfl[f"{t}_upper90"], alpha=0.2)
                plt.xlabel(lv)
                plt.ylabel(t)
                plt.title(f"What-if: {t} vs {lv}")
                plt.tight_layout()
                plt.savefig(out_base / f"whatif_{t}_vs_{lv}.png", dpi=150)
                plt.close()
    except Exception:
        pass

    return str(out_csv)


def try_shap_explanations(
    bundle: ModelBundle,
    X_train_raw: pd.DataFrame,
    *,
    scenario_row: pd.Series,
    outdir: str = "outputs",
    background_n: int = 400,
) -> Optional[str]:
    """
    Optional SHAP explainability.
    If shap is not installed, returns None without failing the pipeline.
    """
    try:
        import shap  # type: ignore
    except Exception:
        return None

    out_base = _ensure_dir(Path(outdir) / "explain_shap")
    X_bg = _sample_df(X_train_raw, n=background_n, seed=RANDOM_STATE)

    # Build one explainer per target and save global + local plots.
    # Note: pipelines include preprocessing; we explain the final XGB model on the transformed feature matrix.
    from scipy import sparse

    def _as_dense(Xt):
        if sparse.issparse(Xt):
            Xt = Xt.tocsr().toarray()
        Xt = np.asarray(Xt)
        if Xt.dtype != np.float32:
            Xt = Xt.astype(np.float32, copy=False)
        return Xt

    def _feature_names(pipe, fallback_dim: int) -> List[str]:
        try:
            pre = pipe.named_steps["preprocess"]
            if hasattr(pre, "get_feature_names_out"):
                return pre.get_feature_names_out().tolist()
        except Exception:
            pass
        return [f"f{i}" for i in range(int(fallback_dim))]

    def _save_summary(shap_vals, X, feature_names, out_path: Path, max_display: int = 20):
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, X, feature_names=feature_names, show=False, max_display=max_display)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    def _save_local(exp, out_path: Path, max_display: int = 20):
        try:
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(exp, max_display=max_display, show=False)
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
        except Exception:
            pass

    scenario_df = scenario_row.to_frame().T

    # Combine artifacts for iteration
    all_targets: List[Tuple[str, Any, str]] = []
    for t, art in bundle.regression.items():
        all_targets.append((t, art.pipeline, "regression"))
    for t, art in bundle.classification.items():
        all_targets.append((t, art.pipeline, "classification"))

    for t, pipe, task in all_targets:
        pre = pipe.named_steps["preprocess"]
        model = pipe.named_steps["model"]
        Xt_bg = _as_dense(pre.transform(X_bg))
        Xt_loc = _as_dense(pre.transform(scenario_df))
        fn = _feature_names(pipe, Xt_bg.shape[1])

        is_clf = task == "classification"
        try:
            expl = shap.TreeExplainer(model, model_output="probability" if is_clf else "raw")
        except Exception:
            expl = shap.TreeExplainer(model)

        shap_bg = expl.shap_values(Xt_bg)
        # Normalize for list outputs
        if isinstance(shap_bg, list):
            shap_bg = shap_bg[-1]
        shap_loc = expl.shap_values(Xt_loc)
        if isinstance(shap_loc, list):
            shap_loc = shap_loc[-1]

        _save_summary(shap_bg, Xt_bg, fn, out_base / f"shap_global_{t}.png")
        try:
            base = expl.expected_value
            if isinstance(base, (list, np.ndarray)):
                base = base[-1]
            exp = shap.Explanation(
                values=np.asarray(shap_loc).reshape(-1),
                base_values=base,
                data=Xt_loc.reshape(-1),
                feature_names=fn,
            )
            _save_local(exp, out_base / f"shap_local_{t}.png")
        except Exception:
            pass

    return str(out_base)