#!/usr/bin/env python3
"""
dollar_backtest.py — simulate a $10,000 starting capital backtest
                     on the saved OOS predictions.

Usage
-----
    python scripts/dollar_backtest.py
    python scripts/dollar_backtest.py --capital 10000 --threshold 0.0002
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import get_logger

logger = get_logger("dollar_backtest")


def run_dollar_backtest(
    oos: pd.DataFrame,
    initial_capital: float = 10_000.0,
    threshold: float = 0.0002,
    cost_per_trade: float = 0.0001,
    slippage: float = 0.00005,
) -> pd.DataFrame:
    bt = oos[["y_true", "y_pred"]].copy()

    bt["position"] = 0
    bt.loc[bt["y_pred"] >  threshold, "position"] =  1
    bt.loc[bt["y_pred"] < -threshold, "position"] = -1

    pos_change = bt["position"].diff().abs()
    pos_change.iloc[0] = abs(bt["position"].iloc[0])
    bt["tc"] = pos_change * (cost_per_trade + slippage)

    bt["strategy_ret"] = bt["position"] * bt["y_true"] - bt["tc"]

    # Equity curve: compound log returns into dollar value
    bt["cum_log_ret"] = bt["strategy_ret"].cumsum()
    bt["equity"] = initial_capital * np.exp(bt["cum_log_ret"])

    # Dollar P&L per bar
    bt["pnl"] = bt["equity"].diff().fillna(bt["equity"].iloc[0] - initial_capital)

    return bt


def print_summary(bt: pd.DataFrame, initial_capital: float, threshold: float,
                  cost_per_trade: float, slippage: float) -> None:
    final_equity  = bt["equity"].iloc[-1]
    total_pnl     = final_equity - initial_capital
    total_ret_pct = (final_equity / initial_capital - 1) * 100
    max_equity    = bt["equity"].max()
    min_equity    = bt["equity"].min()

    running_max   = bt["equity"].cummax()
    drawdown_dol  = bt["equity"] - running_max
    max_dd_dol    = float(drawdown_dol.min())
    dd_pos        = int(drawdown_dol.argmin())
    max_dd_pct    = max_dd_dol / float(running_max.iloc[dd_pos]) * 100

    active        = bt[bt["position"] != 0]
    n_trades      = int((bt["position"].diff().abs() > 0).sum())
    hit_rate      = float((np.sign(active["y_pred"]) == np.sign(active["y_true"])).mean()) if len(active) else float("nan")

    rets          = bt["strategy_ret"]
    sharpe        = float((rets.mean() / rets.std()) * np.sqrt(98_000)) if rets.std() > 0 else float("nan")

    date_start    = bt.index[0]
    date_end      = bt.index[-1]

    print("\n" + "=" * 56)
    print("  DOLLAR BACKTEST — WTI 1-MIN RETURN PREDICTOR")
    print("=" * 56)
    print(f"  Period:            {date_start:%Y-%m-%d} to {date_end:%Y-%m-%d}")
    print(f"  Threshold:         {threshold:.5f}")
    print(f"  Cost + slippage:   {(cost_per_trade + slippage)*1e4:.1f} bps/side")
    print("-" * 56)
    print(f"  Starting capital:  ${initial_capital:>12,.2f}")
    print(f"  Final equity:      ${final_equity:>12,.2f}")
    print(f"  Total P&L:         ${total_pnl:>+12,.2f}   ({total_ret_pct:+.2f}%)")
    print(f"  Peak equity:       ${max_equity:>12,.2f}")
    print(f"  Trough equity:     ${min_equity:>12,.2f}")
    print("-" * 56)
    print(f"  Max drawdown:      ${max_dd_dol:>+12,.2f}   ({max_dd_pct:.2f}%)")
    print(f"  Sharpe (ann.):     {sharpe:>12.3f}")
    print(f"  Hit rate:          {hit_rate:>12.2%}")
    print(f"  N trades:          {n_trades:>12,}")
    print(f"  OOS bars:          {len(bt):>12,}")
    print("=" * 56 + "\n")


def plot_dollar_backtest(bt: pd.DataFrame, initial_capital: float,
                         threshold: float, out_path: Path) -> None:
    # Downsample to daily for plotting — keeps chart fast and readable
    daily = bt["equity"].resample("1D").last().dropna()
    daily_dd = (daily - daily.cummax())

    fig, axes = plt.subplots(3, 1, figsize=(13, 10),
                             gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle(f"Dollar Backtest  |  ${initial_capital:,.0f} starting capital  "
                 f"|  threshold={threshold:.5f}", fontsize=13, fontweight="bold")

    # ── 1. Equity curve (daily) ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot(daily.index, daily.values, color="#0f3460", linewidth=1.2, label="Strategy equity")
    ax.axhline(initial_capital, color="grey", linewidth=0.8, linestyle="--", label="Starting capital")
    daily_max = daily.cummax()
    ax.fill_between(daily.index, daily.values, daily_max.values,
                    where=daily.values < daily_max.values,
                    color="#e94560", alpha=0.25, label="Drawdown")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_ylabel("Portfolio Value (USD)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 2. Drawdown in dollars (daily) ───────────────────────────────────────
    ax2 = axes[1]
    ax2.fill_between(daily_dd.index, daily_dd.values, 0, color="#e94560", alpha=0.6)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.set_ylabel("Drawdown (USD)")
    ax2.grid(True, alpha=0.3)

    # ── 3. Trade markers (only non-flat bars) ────────────────────────────────
    ax3 = axes[2]
    trades = bt[bt["position"] != 0]
    longs  = trades[trades["position"] ==  1]
    shorts = trades[trades["position"] == -1]
    ax3.scatter(longs.index,  [1]  * len(longs),  color="#2ecc71", marker="^", s=60, label="Long",  zorder=3)
    ax3.scatter(shorts.index, [-1] * len(shorts), color="#e94560", marker="v", s=60, label="Short", zorder=3)
    ax3.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax3.set_yticks([-1, 0, 1])
    ax3.set_yticklabels(["Short", "Flat", "Long"])
    ax3.set_ylabel("Trades")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved -> {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--capital",   type=float, default=10_000.0)
    p.add_argument("--threshold", type=float, default=0.0002)
    p.add_argument("--cost",      type=float, default=0.0001)
    p.add_argument("--slippage",  type=float, default=0.00005)
    p.add_argument("--artifacts", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    root   = Path(__file__).resolve().parent.parent
    artdir = Path(args.artifacts) if args.artifacts else root / "data" / "artifacts"

    preds_path = artdir / "oos_predictions.parquet"
    if not preds_path.exists():
        print(f"ERROR: {preds_path} not found. Run the pipeline first.")
        sys.exit(1)

    oos = pd.read_parquet(preds_path)

    bt = run_dollar_backtest(
        oos,
        initial_capital=args.capital,
        threshold=args.threshold,
        cost_per_trade=args.cost,
        slippage=args.slippage,
    )

    print_summary(bt, args.capital, args.threshold, args.cost, args.slippage)

    plot_dollar_backtest(bt, args.capital, args.threshold,
                         artdir / "dollar_backtest.png")
