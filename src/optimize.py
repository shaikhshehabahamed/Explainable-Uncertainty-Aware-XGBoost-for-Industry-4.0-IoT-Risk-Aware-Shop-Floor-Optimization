# src/optimize.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .artifacts import ModelBundle
from .data import CONTEXT_FEATURES, DECISION_LEVERS


@dataclass(frozen=True)
class OptimizationConfig:
    n_candidates: int = 5000
    seed: int = 42

    # Use uncertainty bounds for robust optimization
    robust_objectives: bool = True

    # Constraints (defaults are conservative but practical)
    max_maintenance_prob_upper: float = 0.5
    min_success_prob_lower: float = 0.5

    # If provided, constrain predicted failure (upper) as well
    max_failure_score_upper: Optional[float] = None

    # Objective weights for a single recommended solution (Pareto front is still saved)
    w_energy: float = 1.0
    w_completion: float = 1.0
    w_failure: float = 1.0

    # Quantile bounds to use for candidate sampling (if available in bundle)
    use_bundle_bounds: bool = True


def _sample_levers(bounds: Dict[str, Dict[str, Any]], n: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(int(seed))
    data: Dict[str, Any] = {}
    for lv, b in bounds.items():
        lo = float(b["low"])
        hi = float(b["high"])
        dtype = str(b.get("dtype", "float"))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            # Fallback
            lo, hi = 0.0, 1.0
        if dtype == "int":
            vals = rng.randint(int(lo), int(hi) + 1, size=int(n))
        else:
            vals = rng.uniform(lo, hi, size=int(n))
        data[lv] = vals
    return pd.DataFrame(data)


def _pareto_front_mask(objectives: np.ndarray) -> np.ndarray:
    """Return boolean mask for nondominated points (minimization).

    This implementation is optimized for 1–3 objectives:
      - 1D / 2D: O(n log n)
      - 3D:      O(n log n) using a Fenwick tree (prefix-min query)
    For >3 objectives we fall back to a simple O(n^2) routine.

    Notes:
    - NaN/inf objectives are treated as +inf (worst possible).
    - Dominance uses the standard definition: A dominates B if all objectives <= and at least one is strictly <.
    """
    obj = np.asarray(objectives, dtype=float)
    if obj.ndim != 2:
        raise ValueError("objectives must be a 2D array of shape (n_points, n_objectives)")
    n, m = obj.shape
    if n == 0:
        return np.zeros((0,), dtype=bool)

    # Treat non-finite values as worst-possible so they are unlikely to be selected.
    obj = np.where(np.isfinite(obj), obj, np.inf)

    if m == 1:
        best = float(np.min(obj[:, 0]))
        return (obj[:, 0] == best)

    if m == 2:
        # Sort by objective 0 then objective 1.
        order = np.lexsort((obj[:, 1], obj[:, 0]))
        nd = np.zeros(n, dtype=bool)

        best_o1 = np.inf
        i = 0
        while i < n:
            j = i
            o0 = obj[order[i], 0]
            while j < n and obj[order[j], 0] == o0:
                j += 1

            group = order[i:j]
            min_o1_group = float(np.min(obj[group, 1]))

            for idx in group:
                o1 = float(obj[idx, 1])
                if best_o1 <= o1:
                    # Dominated by a point with smaller o0 and o1 <= current.
                    continue
                if o1 > min_o1_group:
                    # Dominated within the group (same o0, smaller o1 exists).
                    continue
                nd[idx] = True

            best_o1 = min(best_o1, min_o1_group)
            i = j

        return nd

    if m == 3:
        # Sort by objective 0, then objective 1, then objective 2.
        order = np.lexsort((obj[:, 2], obj[:, 1], obj[:, 0]))

        # Coordinate-compress objective 1 for BIT indices.
        uniq_o1 = np.unique(obj[:, 1])
        ranks = np.searchsorted(uniq_o1, obj[:, 1]).astype(int)
        size = int(uniq_o1.size)

        # Fenwick tree for prefix minimum over objective 2.
        bit = np.full((size + 1,), np.inf, dtype=float)

        def bit_update(pos: int, val: float) -> None:
            pos += 1  # 1-indexed
            while pos <= size:
                if val < bit[pos]:
                    bit[pos] = val
                pos += pos & -pos

        def bit_query(pos: int) -> float:
            # min over [0..pos]
            pos += 1
            res = np.inf
            while pos > 0:
                if bit[pos] < res:
                    res = bit[pos]
                pos -= pos & -pos
            return float(res)

        nd = np.zeros(n, dtype=bool)

        i = 0
        while i < n:
            j = i
            o0 = obj[order[i], 0]
            while j < n and obj[order[j], 0] == o0:
                j += 1

            group = order[i:j]

            # First filter: dominated by any previous group (strictly smaller o0).
            candidates: List[int] = []
            for idx in group:
                r = int(ranks[idx])
                best_o2 = bit_query(r)
                if best_o2 <= float(obj[idx, 2]):
                    continue
                candidates.append(int(idx))

            # Second filter: nondominated within the o0-tie group in 2D (o1, o2).
            group_nd: List[int] = []
            if candidates:
                cand = np.asarray(candidates, dtype=int)
                ord2 = cand[np.lexsort((obj[cand, 2], obj[cand, 1]))]

                best_o2_prev = np.inf
                k = 0
                while k < ord2.size:
                    l = k
                    o1 = obj[ord2[k], 1]
                    while l < ord2.size and obj[ord2[l], 1] == o1:
                        l += 1

                    batch = ord2[k:l]
                    min_o2_batch = float(np.min(obj[batch, 2]))

                    for idx2 in batch:
                        o2 = float(obj[idx2, 2])
                        if best_o2_prev <= o2:
                            continue  # dominated by smaller o1
                        if o2 > min_o2_batch:
                            continue  # dominated by same o1, smaller o2
                        nd[int(idx2)] = True
                        group_nd.append(int(idx2))

                    best_o2_prev = min(best_o2_prev, min_o2_batch)
                    k = l

            # Update BIT with nondominated points from this group.
            for idx in group_nd:
                bit_update(int(ranks[idx]), float(obj[idx, 2]))

            i = j

        return nd

    # Fallback for higher dimensions: O(n^2) nondominated check.
    is_nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_nd[i]:
            continue
        dominates_i = np.all(obj <= obj[i], axis=1) & np.any(obj < obj[i], axis=1)
        if np.any(dominates_i):
            is_nd[i] = False
            continue
        dominated_by_i = np.all(obj[i] <= obj, axis=1) & np.any(obj[i] < obj, axis=1)
        is_nd[dominated_by_i] = False
        is_nd[i] = True
    return is_nd

def optimize_decision_levers(
    bundle: ModelBundle,
    *,
    context_row: pd.Series,
    outdir: str = "outputs",
    config: Optional[OptimizationConfig] = None,
) -> Dict[str, Any]:
    """Optimize decision levers for a single context row, using trained models + calibrated uncertainty."""
    cfg = config or OptimizationConfig()
    out_base = Path(outdir) / "optimization"
    out_base.mkdir(parents=True, exist_ok=True)

    # Bounds for sampling
    if cfg.use_bundle_bounds and "lever_bounds" in bundle.metadata:
        bounds = bundle.metadata["lever_bounds"]
    else:
        raise ValueError("No lever bounds available in bundle metadata.")

    n_rand = int(max(1, cfg.n_candidates))
    levers_df = _sample_levers(bounds, n=n_rand, seed=int(cfg.seed))

    # Always include the current scenario lever settings as a baseline candidate (row 0).
    baseline: Dict[str, Any] = {}
    for lv in DECISION_LEVERS:
        v = context_row.get(lv, np.nan)
        try:
            v_f = float(v)
        except Exception:
            v_f = float("nan")
        if not np.isfinite(v_f):
            bb = bounds.get(lv, {"low": 0.0, "high": 1.0})
            v_f = (float(bb["low"]) + float(bb["high"])) / 2.0
        if str(bounds.get(lv, {}).get("dtype", "float")) == "int":
            v_f = int(round(v_f))
        baseline[lv] = v_f

    levers_df = pd.concat([pd.DataFrame([baseline]), levers_df], ignore_index=True)
    levers_df["__baseline__"] = 0
    levers_df.loc[0, "__baseline__"] = 1
    n = int(len(levers_df))

    # Build candidate feature frame: repeat context, plug levers
    ctx = context_row[CONTEXT_FEATURES].to_frame().T
    ctx_rep = pd.concat([ctx] * n, ignore_index=True)
    X_cand = pd.concat([ctx_rep.reset_index(drop=True), levers_df[DECISION_LEVERS].reset_index(drop=True)], axis=1)

    # Predict outcomes (vectorized)
    preds: Dict[str, Any] = {}

    # Regression: mean + bounds
    for t, art in bundle.regression.items():
        mean, lo, hi = art.predict_mean_and_bounds(X_cand)
        preds[f"{t}_mean"] = mean
        preds[f"{t}_lower90"] = lo
        preds[f"{t}_upper90"] = hi

    # Classification: calibrated proba + bounds + label
    for t, art in bundle.classification.items():
        p, lb, ub = art.predict_bounds(X_cand)
        preds[f"{t}_p_cal"] = p
        preds[f"{t}_p_lower"] = lb
        preds[f"{t}_p_upper"] = ub
        preds[f"{t}_label"] = art.predict_label(X_cand)

    pred_df = pd.DataFrame(preds)
    pred_df["baseline"] = levers_df["__baseline__"].to_numpy()

    # Add inputs for traceability
    for c in CONTEXT_FEATURES:
        pred_df[f"ctx_{c}"] = ctx_rep[c].to_numpy()
    for lv in DECISION_LEVERS:
        pred_df[f"lever_{lv}"] = levers_df[lv].to_numpy()

    # Constraints (soft violations + feasible flag)
    deadline = float(context_row.get("Deadline", np.nan))
    # Completion constraint uses upper bound if robust objectives
    completion_key = "Completion_Time_upper90" if cfg.robust_objectives else "Completion_Time_mean"
    completion_pred = pred_df[completion_key].to_numpy(dtype=float)
    if np.isfinite(deadline) and deadline > 0:
        completion_violation = np.maximum(0.0, completion_pred - deadline) / max(1.0, deadline)
    else:
        completion_violation = np.zeros(n, dtype=float)

    maint_key = "Maintenance_Required_p_upper"
    success_key = "Optimization_Success_p_lower"
    maint_pred = pred_df.get(maint_key, pd.Series(np.zeros(n))).to_numpy(dtype=float)
    success_pred = pred_df.get(success_key, pd.Series(np.zeros(n))).to_numpy(dtype=float)

    maint_violation = np.maximum(0.0, maint_pred - float(cfg.max_maintenance_prob_upper))
    success_violation = np.maximum(0.0, float(cfg.min_success_prob_lower) - success_pred)

    if cfg.max_failure_score_upper is not None:
        failure_key = "Predictive_Failure_Score_upper90" if cfg.robust_objectives else "Predictive_Failure_Score_mean"
        failure_pred = pred_df[failure_key].to_numpy(dtype=float)
        failure_violation = np.maximum(0.0, failure_pred - float(cfg.max_failure_score_upper))
    else:
        failure_violation = np.zeros(n, dtype=float)

    constraint_violation = completion_violation + maint_violation + success_violation + failure_violation
    feasible = constraint_violation <= 1e-12
    pred_df["feasible"] = feasible.astype(int)
    pred_df["constraint_violation"] = constraint_violation
    pred_df["completion_violation"] = completion_violation
    pred_df["maintenance_violation"] = maint_violation
    pred_df["success_violation"] = success_violation
    pred_df["failure_violation"] = failure_violation

    # Objectives (minimize)
    energy_col = "Energy_Used_upper90" if cfg.robust_objectives else "Energy_Used_mean"
    completion_col = completion_key
    failure_col = "Predictive_Failure_Score_upper90" if cfg.robust_objectives else "Predictive_Failure_Score_mean"

    obj = pred_df[[energy_col, completion_col, failure_col]].to_numpy(dtype=float)
    # Penalize by soft constraint violations (keeps ranking meaningful even if no strictly feasible solution exists)
    penalty = constraint_violation.astype(float) * 1000.0
    obj_pen = obj + penalty[:, None]

    # Pareto front among feasible
    feasible_idx = np.where(feasible)[0]
    pareto_note = "feasible"
    if feasible_idx.size > 0:
        nd_mask_feas = _pareto_front_mask(obj[feasible_idx])
        pareto_idx = feasible_idx[nd_mask_feas]
    else:
        # No strictly feasible solutions — take a "near-feasible" Pareto set among the least-violating candidates.
        pareto_note = "near_feasible"
        k = int(max(1, round(0.05 * n)))
        idx_small = np.argsort(constraint_violation)[:k]
        nd_mask_nf = _pareto_front_mask(obj_pen[idx_small])
        pareto_idx = idx_small[nd_mask_nf]

    pred_df["pareto"] = 0
    pred_df.loc[pareto_idx, "pareto"] = 1

    # Choose a single recommended solution by weighted normalized score
    if pareto_idx.size > 0:
        pareto_obj = obj_pen[pareto_idx]  # penalties are 0 for feasible points
        # Normalize each objective by min/max on pareto set
        mins = pareto_obj.min(axis=0)
        maxs = pareto_obj.max(axis=0)
        denom = np.where(maxs > mins, (maxs - mins), 1.0)
        norm = (pareto_obj - mins) / denom
        w = np.array([cfg.w_energy, cfg.w_completion, cfg.w_failure], dtype=float)
        score = norm @ w
        best_local = int(np.argmin(score))
        best_idx = int(pareto_idx[best_local])
    else:
        # Fallback: best penalized objective sum
        w = np.array([cfg.w_energy, cfg.w_completion, cfg.w_failure], dtype=float)
        score = (obj_pen @ w).astype(float)
        best_idx = int(np.argmin(score))

    pred_df["score"] = (obj_pen @ np.array([cfg.w_energy, cfg.w_completion, cfg.w_failure], dtype=float)).astype(float)
    pred_df["recommended"] = 0
    pred_df.loc[best_idx, "recommended"] = 1

    # Save outputs
    solutions_path = out_base / "solutions.csv"
    pred_df.to_csv(solutions_path, index=False)

    pareto_path = out_base / "pareto_front.csv"
    if pareto_idx.size > 0:
        pred_df.loc[pareto_idx].to_csv(pareto_path, index=False)
    else:
        pd.DataFrame().to_csv(pareto_path, index=False)

    best_row = pred_df.loc[best_idx].to_dict()
    with open(out_base / "recommended_solution.json", "w", encoding="utf-8") as f:
        json.dump(best_row, f, indent=2)

    # Plot: energy vs completion with pareto highlight (if possible)
    try:
        plt.figure()
        plt.scatter(pred_df[energy_col], pred_df[completion_col], s=10, alpha=0.25)
        if pareto_idx.size > 0:
            plt.scatter(pred_df.loc[pareto_idx, energy_col], pred_df.loc[pareto_idx, completion_col], s=18, alpha=0.9)
        plt.scatter([pred_df.loc[best_idx, energy_col]], [pred_df.loc[best_idx, completion_col]], s=40, marker="x")
        plt.xlabel(energy_col)
        plt.ylabel(completion_col)
        plt.title("Optimization candidates (Pareto highlighted)")
        plt.tight_layout()
        plt.savefig(out_base / "pareto_energy_vs_completion.png", dpi=150)
        plt.close()
    except Exception:
        pass

    return {
        "solutions_csv": str(solutions_path),
        "pareto_csv": str(pareto_path),
        "recommended_json": str(out_base / "recommended_solution.json"),
        "n_candidates": int(n),
        "n_feasible": int(np.sum(feasible)),
        "n_pareto": int(pareto_idx.size),
        "recommended_index": int(best_idx),
        "pareto_set_type": pareto_note,
        "baseline_index": 0,
    }