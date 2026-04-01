#!/usr/bin/env python3
"""
run_pipeline.py — end-to-end pipeline runner.

Usage
-----
    python scripts/run_pipeline.py                        # uses default config
    python scripts/run_pipeline.py --config path/to.yaml
    python scripts/run_pipeline.py --cv rolling           # rolling-window CV

Steps
-----
1. Load config
2. Load + clean OHLCV data
3. Build features and target
4. Run walk-forward validation (fit XGBoost, collect OOS predictions)
5. Compute regression metrics
6. Run backtest + threshold sweep
7. Generate plots
8. Print summary
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make sure the project root is on the path regardless of where the script is called from
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.backtest import compute_backtest_stats, run_backtest, threshold_sweep
from src.data_loader import load_and_clean
from src.feature_engineering import build_features, get_feature_names
from src.metrics import evaluate_predictions
from src.plotting import (
    plot_actual_vs_predicted,
    plot_cumulative_returns,
    plot_feature_importance,
    plot_fold_metrics,
    plot_prediction_histogram,
    plot_threshold_sweep,
)
from src.train import run_walk_forward
from src.utils import ensure_dir, get_logger, load_config, resolve_path
from src.validation import make_cv

logger = get_logger("pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oil 1-min return prediction pipeline")
    parser.add_argument(
        "--config", default=None,
        help="Path to config YAML (default: config/config.yaml)"
    )
    parser.add_argument(
        "--cv", default="expanding", choices=["expanding", "rolling"],
        help="Walk-forward CV strategy (default: expanding)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    # ── 1. Config ─────────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    artifacts_dir = ensure_dir(cfg["output"]["artifacts_dir"])
    logger.info("Pipeline start | artifacts → %s", artifacts_dir)

    # ── 2. Data ───────────────────────────────────────────────────────────────
    logger.info("=== Step 1: Load & clean data ===")
    ohlcv = load_and_clean(cfg)
    logger.info("OHLCV: %d bars | %s → %s", len(ohlcv), ohlcv.index.min(), ohlcv.index.max())

    # ── 3. Features ───────────────────────────────────────────────────────────
    logger.info("=== Step 2: Feature engineering ===")
    model_df = build_features(ohlcv, cfg["features"])
    feature_names = get_feature_names(model_df)
    logger.info("%d features | %d modelling rows", len(feature_names), len(model_df))

    # ── 4. Walk-forward validation ────────────────────────────────────────────
    logger.info("=== Step 3: Walk-forward validation (%s window) ===", args.cv)
    cv = make_cv(cfg["validation"], strategy=args.cv)
    n_folds = cv.n_splits(model_df)
    logger.info("Estimated folds: %d", n_folds)

    oos = run_walk_forward(
        model_df=model_df,
        cv=cv,
        model_cfg=cfg["model"],
        output_cfg=cfg["output"],
    )
    logger.info("OOS predictions: %d rows | %d folds", len(oos), oos["fold"].nunique())

    # ── 5. Metrics ────────────────────────────────────────────────────────────
    logger.info("=== Step 4: Regression metrics ===")
    metrics = evaluate_predictions(oos, metrics_path=cfg["output"]["metrics_path"])

    # ── 6. Backtest ───────────────────────────────────────────────────────────
    logger.info("=== Step 5: Backtest ===")
    bt_cfg = cfg["backtest"]
    bt = run_backtest(
        oos,
        threshold=bt_cfg["threshold"],
        cost_per_trade=bt_cfg["cost_per_trade"],
        slippage=bt_cfg["slippage"],
        annualise_factor=bt_cfg["annualise_factor"],
    )
    bt_stats = compute_backtest_stats(bt, annualise_factor=bt_cfg["annualise_factor"])

    logger.info(
        "Backtest (threshold=%.5f) | Sharpe=%.3f | Total ret=%.4f | MaxDD=%.4f "
        "| HitRate=%.3f | Trades=%d",
        bt_cfg["threshold"],
        bt_stats["annualised_sharpe"],
        bt_stats["total_log_return"],
        bt_stats["max_drawdown"],
        bt_stats["hit_rate"],
        bt_stats["n_trades"],
    )

    # Threshold sweep
    logger.info("=== Step 5b: Threshold sweep ===")
    sweep = threshold_sweep(
        oos,
        thresholds=bt_cfg["threshold_sweep"],
        cost_per_trade=bt_cfg["cost_per_trade"],
        slippage=bt_cfg["slippage"],
        annualise_factor=bt_cfg["annualise_factor"],
    )
    sweep.to_csv(artifacts_dir / "threshold_sweep.csv")

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    logger.info("=== Step 6: Plots ===")
    plot_actual_vs_predicted(oos, artifacts_dir)
    plot_cumulative_returns(bt, artifacts_dir)
    plot_prediction_histogram(oos, threshold=bt_cfg["threshold"], artifacts_dir=artifacts_dir)
    plot_fold_metrics(metrics["per_fold"], artifacts_dir)
    plot_threshold_sweep(sweep, artifacts_dir)

    # Feature importance plot (uses last-fold model saved by train.py)
    model_path = resolve_path(cfg["output"]["artifacts_dir"]) / "model_last_fold.pkl"
    feat_path = resolve_path(cfg["output"]["artifacts_dir"]) / "feature_names.json"
    if model_path.exists() and feat_path.exists():
        with feat_path.open() as fh:
            saved_feature_names = json.load(fh)
        plot_feature_importance(model_path, saved_feature_names, artifacts_dir)

    # ── 8. Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    overall = metrics["overall"]
    print("\n" + "=" * 60)
    print("  OIL 1-MIN RETURN PREDICTOR — PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Data bars:         {len(ohlcv):>10,}")
    print(f"  Modelling rows:    {len(model_df):>10,}")
    print(f"  Features:          {len(feature_names):>10,}")
    print(f"  CV folds:          {oos['fold'].nunique():>10,}")
    print(f"  OOS predictions:   {len(oos):>10,}")
    print()
    print("  Regression metrics (OOS)")
    print(f"    RMSE:            {overall['rmse']:>12.6f}")
    print(f"    MAE:             {overall['mae']:>12.6f}")
    print(f"    Correlation:     {overall['corr']:>12.4f}")
    print(f"    Sign accuracy:   {overall['sign_accuracy']:>12.4f}")
    print()
    print(f"  Backtest (threshold={bt_cfg['threshold']:.5f})")
    print(f"    Sharpe (ann.):   {bt_stats['annualised_sharpe']:>12.3f}")
    print(f"    Total log ret:   {bt_stats['total_log_return']:>12.4f}")
    print(f"    Max drawdown:    {bt_stats['max_drawdown']:>12.4f}")
    print(f"    Hit rate:        {bt_stats['hit_rate']:>12.4f}")
    print(f"    N trades:        {bt_stats['n_trades']:>12,}")
    print()
    print(f"  Elapsed:           {elapsed:>10.1f}s")
    print(f"  Artifacts:         {artifacts_dir}")
    print("=" * 60)
    print()
    print("  CAUTION: 1-min returns are noisy. Predictive power may be")
    print("  very small.  Transaction costs can erase apparent edge.")
    print("  This is research code, not production trading advice.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
