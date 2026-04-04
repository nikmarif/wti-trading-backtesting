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
# 6. Strategy trade chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_strategy_trades(
    bt: pd.DataFrame,
    strategy_name: str,
    artifacts_dir: str | Path,
    max_bars: int = 500,
    prices: pd.Series | None = None,
    indicators: dict | None = None,
) -> Path:
    """
    Two-panel chart:
      - Top: price with shaded holding regions, entry markers, P&L labels, and
             optional indicator overlays (MAs, bands, etc.).
      - Bottom: cumulative equity curve over the same window.

    Parameters
    ----------
    bt : pd.DataFrame
        Output of run_backtest_from_positions(). Must have columns:
        y_true, position, strategy_ret, cumulative_ret.
        Index should be a DatetimeIndex (used as x-axis).
    strategy_name : str
        Used in the chart title.
    artifacts_dir : str or Path
        Directory to save the PNG.
    max_bars : int
        Number of bars to display (first max_bars rows of bt).
    prices : pd.Series, optional
        Original close price series. If provided, used directly instead of
        reconstructing from log returns.
    indicators : dict[str, pd.Series], optional
        Extra lines to overlay on the price panel, e.g. {"Fast MA (5)": series}.
        Series should share the same index as prices/bt.
    """
    artifacts_dir = ensure_dir(artifacts_dir)
    window = bt.iloc[:max_bars].copy()

    # Price — use real prices if provided, otherwise reconstruct from log returns
    if prices is not None:
        price_window = prices.reindex(window.index)
        price = price_window.values
    else:
        base = 100.0
        log_price = np.concatenate([[0.0], window["y_true"].cumsum().values[:-1]])
        price = base * np.exp(log_price)

    # ── Identify trades ───────────────────────────────────────────────────────
    # A trade is a contiguous run of non-zero position.
    pos = window["position"]
    trades = []
    in_trade = False
    entry_idx = None
    entry_price = None
    direction = 0

    for i, (ts, p) in enumerate(pos.items()):
        if not in_trade and p != 0:
            in_trade = True
            entry_idx = i
            entry_price = price[i]
            direction = p
        elif in_trade and (p == 0 or p != direction):
            # Close the trade at bar i-1
            exit_idx = i - 1
            pnl = window["strategy_ret"].iloc[entry_idx:exit_idx + 1].sum()
            trades.append({
                "entry_i":     entry_idx,
                "exit_i":      exit_idx,
                "entry_ts":    window.index[entry_idx],
                "exit_ts":     window.index[exit_idx],
                "entry_price": entry_price,
                "exit_price":  price[exit_idx],
                "direction":   direction,
                "pnl":         pnl,
            })
            in_trade = False
            # Immediately open a new trade if new position is non-zero
            if p != 0:
                in_trade = True
                entry_idx = i
                entry_price = price[i]
                direction = p

    # Close any open trade at end of window
    if in_trade:
        exit_idx = len(window) - 1
        pnl = window["strategy_ret"].iloc[entry_idx:exit_idx + 1].sum()
        trades.append({
            "entry_i":     entry_idx,
            "exit_i":      exit_idx,
            "entry_ts":    window.index[entry_idx],
            "exit_ts":     window.index[exit_idx],
            "entry_price": entry_price,
            "exit_price":  price[exit_idx],
            "direction":   direction,
            "pnl":         pnl,
        })

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax_price, ax_equity) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=False,
    )
    x = np.arange(len(window))
    xtick_step = max(1, len(window) // 10)
    xtick_pos = x[::xtick_step]
    xtick_labels = [window.index[i].strftime("%m-%d %H:%M") for i in xtick_pos]

    # ── Top panel: price + trades ─────────────────────────────────────────────
    price_label = "Price" if prices is not None else "Price (rebased)"
    ax_price.plot(x, price, color="#333333", lw=0.9, zorder=2, label=price_label)

    # Indicator overlays (MAs, bands, etc.)
    indicator_colors = ["#e67e22", "#8e44ad", "#2980b9", "#16a085"]
    for (ind_label, ind_series), color in zip((indicators or {}).items(), indicator_colors):
        ind_window = ind_series.reindex(window.index).values
        ax_price.plot(x, ind_window, lw=0.9, color=color, zorder=3, label=ind_label)

    for t in trades:
        ei, xi = t["entry_i"], t["exit_i"]
        color = "#2ecc71" if t["direction"] == 1 else "#e74c3c"
        light = "#d5f5e3" if t["direction"] == 1 else "#fadbd8"

        # Shaded holding region
        ax_price.axvspan(ei, xi, color=light, alpha=0.6, zorder=1)

        # Entry marker
        marker = "^" if t["direction"] == 1 else "v"
        ax_price.scatter(
            ei, price[ei], marker=marker, color=color,
            s=60, zorder=4, linewidths=0,
        )

        # Exit marker (circle)
        ax_price.scatter(
            xi, price[xi], marker="o", color=color,
            s=40, zorder=4, linewidths=0,
        )

        # P&L label at exit — only if trade is fully within window
        pnl_pct = t["pnl"] * 100
        label = f"{'+' if pnl_pct >= 0 else ''}{pnl_pct:.3f}%"
        label_color = "#1a8a4a" if pnl_pct >= 0 else "#c0392b"
        ax_price.annotate(
            label,
            xy=(xi, price[xi]),
            xytext=(0, 10 if t["direction"] == 1 else -14),
            textcoords="offset points",
            fontsize=6.5,
            color=label_color,
            ha="center",
            zorder=5,
        )

    ax_price.set_title(
        f"{strategy_name} — Price with Trades  (first {len(window)} bars)",
        fontsize=11, fontweight="bold",
    )
    ax_price.set_ylabel("Price (rebased to 100)")
    ax_price.set_xticks(xtick_pos)
    ax_price.set_xticklabels(xtick_labels, rotation=30, ha="right", fontsize=7)

    # Legend elements
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], color="#333333", lw=1, label="Price"),
        Patch(facecolor="#d5f5e3", edgecolor="none", label="Long"),
        Patch(facecolor="#fadbd8", edgecolor="none", label="Short"),
    ]
    ax_price.legend(handles=legend_els, fontsize=8, loc="upper left")

    # ── Bottom panel: equity curve ────────────────────────────────────────────
    cum = window["cumulative_ret"].values
    ax_equity.plot(x, cum, color="steelblue", lw=0.9)
    ax_equity.axhline(0, color="k", lw=0.5, linestyle="--")
    ax_equity.fill_between(x, cum, 0,
                           where=(cum >= 0), color="#d5f5e3", alpha=0.7)
    ax_equity.fill_between(x, cum, 0,
                           where=(cum < 0),  color="#fadbd8", alpha=0.7)
    ax_equity.set_ylabel("Cum. log return")
    ax_equity.set_xlabel("Bar")
    ax_equity.set_xticks(xtick_pos)
    ax_equity.set_xticklabels(xtick_labels, rotation=30, ha="right", fontsize=7)

    fig.suptitle(
        f"Backtest — {strategy_name}",
        fontsize=13, fontweight="bold", y=1.01,
    )

    fname = f"trades_{strategy_name.lower().replace(' ', '_')}.png"
    return _save(fig, artifacts_dir / fname)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Threshold sweep table plot
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
