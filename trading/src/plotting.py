"""
plotting.py — save figures to the artifacts directory.

All functions save to disk and return the file path.  They do not call
plt.show() so the pipeline can run headlessly.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless backend — must be set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .utils import ensure_dir, get_logger

logger = get_logger(__name__)
plt.rcParams.update({"figure.dpi": 120, "font.size": 9})


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot → %s", path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 1. Actual vs predicted scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(
    oos: pd.DataFrame,
    artifacts_dir: str | Path,
    sample_n: int = 5000,
) -> Path:
    """Scatter of y_true vs y_pred for a random time-contiguous sample."""
    artifacts_dir = ensure_dir(artifacts_dir)
    sample = oos.iloc[:sample_n] if len(oos) > sample_n else oos

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Scatter
    ax = axes[0]
    ax.scatter(sample["y_pred"], sample["y_true"], alpha=0.15, s=4, color="steelblue")
    lim = max(abs(sample["y_true"].max()), abs(sample["y_true"].min()),
               abs(sample["y_pred"].max()), abs(sample["y_pred"].min())) * 1.1
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=0.8, label="y=x")
    ax.set_xlabel("Predicted log return")
    ax.set_ylabel("Actual log return")
    ax.set_title("Actual vs Predicted (first %d OOS bars)" % len(sample))
    ax.legend()

    # Time-series line (predicted vs actual)
    ax2 = axes[1]
    ax2.plot(sample.index, sample["y_true"], lw=0.5, alpha=0.7, label="Actual")
    ax2.plot(sample.index, sample["y_pred"], lw=0.5, alpha=0.7, label="Predicted")
    ax2.set_title("Returns over time")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Log return")
    ax2.legend()

    return _save(fig, artifacts_dir / "actual_vs_predicted.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cumulative strategy returns
# ─────────────────────────────────────────────────────────────────────────────

def plot_cumulative_returns(
    bt: pd.DataFrame,
    artifacts_dir: str | Path,
) -> Path:
    """Plot cumulative strategy return vs buy-and-hold."""
    artifacts_dir = ensure_dir(artifacts_dir)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(bt.index, bt["cumulative_ret"], label="Strategy (log ret)", lw=0.8)

    # Buy-and-hold: cumulative sum of y_true
    bh = bt["y_true"].cumsum()
    ax.plot(bt.index, bh, label="Buy-and-hold (log ret)", lw=0.8, alpha=0.7,
            linestyle="--")

    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Cumulative Log Return — Strategy vs Buy-and-Hold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative log return")
    ax.legend()

    return _save(fig, artifacts_dir / "cumulative_returns.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Prediction histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_prediction_histogram(
    oos: pd.DataFrame,
    threshold: float,
    artifacts_dir: str | Path,
) -> Path:
    """Histogram of predicted values with threshold lines."""
    artifacts_dir = ensure_dir(artifacts_dir)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(oos["y_pred"], bins=100, color="steelblue", alpha=0.7, edgecolor="none")
    ax.axvline( threshold, color="green", lw=1.2, linestyle="--", label=f"+{threshold}")
    ax.axvline(-threshold, color="red",   lw=1.2, linestyle="--", label=f"-{threshold}")
    ax.set_title("Distribution of OOS Predictions")
    ax.set_xlabel("Predicted log return")
    ax.set_ylabel("Count")
    ax.legend()

    return _save(fig, artifacts_dir / "prediction_histogram.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Feature importance
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    model_path: str | Path,
    feature_names: list[str],
    artifacts_dir: str | Path,
    top_n: int = 30,
) -> Path:
    """Bar chart of XGBoost feature importances from the last-fold model."""
    artifacts_dir = ensure_dir(artifacts_dir)

    with open(model_path, "rb") as fh:
        model = pickle.load(fh)

    importance = pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    importance.plot.barh(ax=ax, color="steelblue", edgecolor="none")
    ax.set_title(f"XGBoost Feature Importance (top {top_n})")
    ax.set_xlabel("Importance (gain)")

    return _save(fig, artifacts_dir / "feature_importance.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Per-fold performance summary
# ─────────────────────────────────────────────────────────────────────────────

def plot_fold_metrics(
    per_fold_metrics: dict,
    artifacts_dir: str | Path,
) -> Path:
    """Bar chart of per-fold correlation and sign accuracy."""
    artifacts_dir = ensure_dir(artifacts_dir)

    folds = sorted(per_fold_metrics.keys())
    corrs = [per_fold_metrics[f]["corr"] for f in folds]
    sign_accs = [per_fold_metrics[f]["sign_accuracy"] for f in folds]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(folds, corrs, color="steelblue", edgecolor="none")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_title("Per-fold Pearson Correlation")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("Correlation")

    axes[1].bar(folds, sign_accs, color="seagreen", edgecolor="none")
    axes[1].axhline(0.5, color="r", lw=0.8, linestyle="--", label="Random (0.5)")
    axes[1].set_title("Per-fold Sign Accuracy")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("Sign accuracy")
    axes[1].legend()

    return _save(fig, artifacts_dir / "fold_metrics.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Threshold sweep table plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_threshold_sweep(
    sweep_df: pd.DataFrame,
    artifacts_dir: str | Path,
) -> Path:
    """Line plots of Sharpe and total return vs threshold."""
    artifacts_dir = ensure_dir(artifacts_dir)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    thr = sweep_df.index.values

    axes[0].plot(thr, sweep_df["annualised_sharpe"], marker="o", color="steelblue")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_title("Annualised Sharpe vs Threshold")
    axes[0].set_xlabel("Threshold (log return)")
    axes[0].set_ylabel("Sharpe")
    axes[0].xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    axes[1].plot(thr, sweep_df["total_log_return"], marker="o", color="seagreen")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_title("Total Log Return vs Threshold")
    axes[1].set_xlabel("Threshold (log return)")
    axes[1].set_ylabel("Total log return")

    return _save(fig, artifacts_dir / "threshold_sweep.png")
