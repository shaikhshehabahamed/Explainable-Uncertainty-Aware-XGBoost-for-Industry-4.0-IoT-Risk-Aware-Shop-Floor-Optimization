# main.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.artifacts import ModelBundle
from src.data import extract_features_for_inference, read_dataset, validate_exact_columns, drop_setpoint_constant
from src.pipeline import prepare_data
from src.training import TrainingConfig, evaluate_on_test, train_all_targets
from src.optimize import OptimizationConfig, optimize_decision_levers
from src.explain import permutation_importance, try_shap_explanations, what_if_lever_sensitivity


def _save_json(obj: Any, path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return str(p)


def cmd_train(args: argparse.Namespace) -> None:
    data = prepare_data(args.data, strict_schema=True, seed=int(args.seed))

    cfg = TrainingConfig(
        device=args.device,
        seed=args.seed,
        n_jobs=args.n_jobs,
        max_bin=args.max_bin,
        tree_method=args.tree_method,
        n_splits=args.cv_splits,
        early_stopping_rounds=args.early_stopping_rounds,
        early_stopping_test_size=args.early_stopping_test_size,
        pi_alpha=args.pi_alpha,
        clf_bounds_alpha=args.clf_bounds_alpha,
        clf_bounds_bins=args.clf_bounds_bins,
        threshold_metric=args.threshold_metric,
        hpo_trials=args.hpo_trials,
        hpo_seed=args.hpo_seed,
        hpo_max_rows=args.hpo_max_rows,
    )

    bundle, train_summary = train_all_targets(
        data.X_train_raw,
        data.y_train,
        data.groups_train,
        outdir=args.outdir,
        config=cfg,
    )

    test_summary = evaluate_on_test(bundle, data.X_test_raw, data.y_test, outdir=args.outdir)

    out = {
        "train_summary_path": str(Path(args.outdir) / "metrics" / "train_summary.json"),
        "test_summary_path": str(Path(args.outdir) / "metrics" / "test_summary.json"),
        "bundle_path": str(Path(args.outdir) / "models" / "bundle.joblib"),
    }
    _save_json(out, str(Path(args.outdir) / "run_manifest.json"))

    if train_summary.get("by_target") and isinstance(train_summary["by_target"], dict):
        # Print a minimal confirmation to stdout (no long logs)
        print(f"Training complete. Bundle saved to: {out['bundle_path']}")
        if bundle.metadata.get("device_warning"):
            print(f"NOTE: {bundle.metadata.get('device_warning')}")
        print(f"Test metrics saved to: {Path(args.outdir) / 'metrics' / 'test_metrics.csv'}")


def cmd_predict(args: argparse.Namespace) -> None:
    bundle = ModelBundle.load(args.bundle)

    df = pd.read_csv(args.input)
    # Allow both full-schema dataset and feature-only inference files
    X = extract_features_for_inference(df, strict_setpoint=False)

    out_rows: Dict[str, Any] = {}

    # Keep useful identifiers if present
    if "Unit_ID" in df.columns:
        out_rows["Unit_ID"] = df["Unit_ID"].to_numpy()

    # Regression predictions
    for t, art in bundle.regression.items():
        mean, lo, hi = art.predict_mean_and_bounds(X)
        out_rows[f"{t}_pred"] = mean
        out_rows[f"{t}_lower90"] = lo
        out_rows[f"{t}_upper90"] = hi

    # Classification predictions
    for t, art in bundle.classification.items():
        p, lb, ub = art.predict_bounds(X)
        out_rows[f"{t}_p_cal"] = p
        out_rows[f"{t}_p_lower"] = lb
        out_rows[f"{t}_p_upper"] = ub
        out_rows[f"{t}_label"] = art.predict_label(X)

    out_df = pd.DataFrame(out_rows)

    if args.include_inputs:
        # Prefix input columns to avoid collisions
        for c in X.columns:
            out_df[f"input_{c}"] = X[c].to_numpy()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Predictions written to: {out_path}")


def cmd_optimize(args: argparse.Namespace) -> None:
    bundle = ModelBundle.load(args.bundle)

    df = read_dataset(args.data)
    # Strict schema here because we rely on known context columns
    validate_exact_columns(df)
    df = drop_setpoint_constant(df, strict=True)

    # Extract model features and select a scenario row
    X = extract_features_for_inference(df, strict_setpoint=False)
    if args.row_index < 0 or args.row_index >= len(X):
        raise IndexError(f"--row-index out of range: {args.row_index} (n_rows={len(X)})")

    scenario_row = X.iloc[int(args.row_index)]

    cfg = OptimizationConfig(
        n_candidates=args.n_candidates,
        seed=args.seed,
        robust_objectives=not args.non_robust,
        max_maintenance_prob_upper=args.max_maint_upper,
        min_success_prob_lower=args.min_success_lower,
        max_failure_score_upper=args.max_failure_upper,
        w_energy=args.w_energy,
        w_completion=args.w_completion,
        w_failure=args.w_failure,
    )

    result = optimize_decision_levers(bundle, context_row=scenario_row, outdir=args.outdir, config=cfg)
    print(json.dumps(result, indent=2))


def cmd_explain(args: argparse.Namespace) -> None:
    # Explain requires training data (for permutation importance background)
    data = prepare_data(args.data, strict_schema=True, seed=int(args.seed))
    bundle = ModelBundle.load(args.bundle)

    # Global importance (works without SHAP)
    csv_path = permutation_importance(bundle, data.X_train_raw, data.y_train, outdir=args.outdir)

    # Scenario row for local analysis
    if args.row_index < 0 or args.row_index >= len(data.X_train_raw):
        row_idx = 0
    else:
        row_idx = int(args.row_index)
    scenario_row = data.X_train_raw.iloc[row_idx]

    whatif_path = what_if_lever_sensitivity(bundle, scenario_row=scenario_row, outdir=args.outdir)

    # Optional SHAP outputs if shap is installed
    shap_dir = try_shap_explanations(bundle, data.X_train_raw, scenario_row=scenario_row, outdir=args.outdir)

    print("Explainability outputs:")
    print(f"  Permutation importance CSV: {csv_path}")
    print(f"  What-if curves CSV:         {whatif_path}")
    if shap_dir:
        print(f"  SHAP plots directory:       {shap_dir}")
    else:
        print("  SHAP not installed; skipped SHAP plots.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Distributed manufacturing ML pipeline (train / predict / optimize / explain).")
    p.add_argument("--data", type=str, default="./distributed_manufacturing_dataset.csv", help="Path to CSV dataset.")
    p.add_argument("--outdir", type=str, default="outputs", help="Output directory root.")
    p.add_argument("--bundle", type=str, default="outputs/models/bundle.joblib", help="Path to saved ModelBundle joblib.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--device", type=str, default="auto", help="XGBoost device: cpu, cuda, cuda:0, or auto.")
    p.add_argument("--n-jobs", dest="n_jobs", type=int, default=4, help="XGBoost n_jobs.")
    p.add_argument("--max-bin", dest="max_bin", type=int, default=256, help="XGBoost max_bin.")
    p.add_argument("--tree-method", dest="tree_method", type=str, default="hist", help="XGBoost tree_method.")

    sub = p.add_subparsers(dest="command")

    # train
    t = sub.add_parser("train", help="Train models, calibrate uncertainty, and evaluate on test split.")
    t.add_argument("--cv-splits", type=int, default=5, help="Max GroupKFold splits (min(5, n_groups)).")
    t.add_argument("--early-stopping-rounds", type=int, default=100, help="Early stopping rounds inside folds.")
    t.add_argument("--early-stopping-test-size", type=float, default=0.2, help="Inner ES split fraction (group-aware).")
    t.add_argument("--pi-alpha", type=float, default=0.1, help="Regression PI miscoverage (0.1 => 90% PI).")
    t.add_argument("--clf-bounds-alpha", type=float, default=0.1, help="Wilson bounds alpha (two-sided).")
    t.add_argument("--clf-bounds-bins", type=int, default=10, help="Number of probability bins for bounds.")
    t.add_argument("--threshold-metric", type=str, default="f1", choices=["f1", "balanced_accuracy", "youden"], help="Threshold selection metric.")
    t.add_argument("--hpo-trials", type=int, default=0, help="Random-search trials per target (0 disables).")
    t.add_argument("--hpo-seed", type=int, default=123, help="HPO RNG seed.")
    t.add_argument("--hpo-max-rows", type=int, default=2500, help="Row subsample size during HPO for speed.")
    t.set_defaults(func=cmd_train)

    # predict
    pr = sub.add_parser("predict", help="Inference mode: load trained bundle and predict on a CSV.")
    pr.add_argument("--input", type=str, required=True, help="Input CSV (feature-only or full-schema).")
    pr.add_argument("--output", type=str, default="outputs/predictions/predictions.csv", help="Output predictions CSV.")
    pr.add_argument("--include-inputs", action="store_true", help="Include input feature columns in output.")
    pr.set_defaults(func=cmd_predict)

    # optimize
    o = sub.add_parser("optimize", help="Optimize decision levers for one scenario row.")
    o.add_argument("--row-index", type=int, default=0, help="Row index from dataset used as the context scenario.")
    o.add_argument("--n-candidates", type=int, default=5000, help="Number of random candidate lever settings.")
    o.add_argument("--non-robust", action="store_true", help="Use mean predictions instead of uncertainty bounds for objectives/constraints.")
    o.add_argument("--max-maint-upper", type=float, default=0.5, help="Constraint: Maintenance_Required upper bound <= value.")
    o.add_argument("--min-success-lower", type=float, default=0.5, help="Constraint: Optimization_Success lower bound >= value.")
    o.add_argument("--max-failure-upper", type=float, default=None, help="Optional constraint: Predictive_Failure_Score upper <= value.")
    o.add_argument("--w-energy", type=float, default=1.0, help="Weight for energy objective.")
    o.add_argument("--w-completion", type=float, default=1.0, help="Weight for completion objective.")
    o.add_argument("--w-failure", type=float, default=1.0, help="Weight for failure-risk objective.")
    o.set_defaults(func=cmd_optimize)

    # explain
    e = sub.add_parser("explain", help="Generate explainability artifacts (permutation importance, what-if, optional SHAP).")
    e.add_argument("--row-index", type=int, default=0, help="Scenario row index from training split.")
    e.set_defaults(func=cmd_explain)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Backward-compatible default: if no subcommand given, run train
    if not getattr(args, "command", None):
        args.command = "train"
        args.func = cmd_train

    args.func(args)


if __name__ == "__main__":
    main()