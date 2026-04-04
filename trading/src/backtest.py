"""
backtest.py — threshold-based backtest on OOS predictions only.

Trading rule
------------
  - At the end of bar t, we observe y_pred[t].
  - If y_pred[t] >  threshold → go long  (+1)
  - If y_pred[t] < -threshold → go short (-1)
  - Otherwise                 → flat     (0)
  - The position is held for ONE bar.
  - The realised PnL at bar t = position[t] * y_true[t]
    where y_true[t] = log(close[t+1]/close[t]).

Transaction costs
-----------------
  A cost is incurred whenever position changes.
  cost[t] = |position[t] - position[t-1]| * (cost_per_trade + slippage)

No leakage: position[t] is based solely on y_pred[t] (computed from
information available at or before bar t), and y_true[t] is the realised
next-bar return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core backtest
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    oos: pd.DataFrame,
    threshold: float = 0.0002,
    cost_per_trade: float = 0.0001,
    slippage: float = 0.00005,
    annualise_factor: int = 98_000,
) -> pd.DataFrame:
    """
    Run the simple threshold backtest on OOS predictions.

    Parameters
    ----------
    oos : pd.DataFrame
        Must have columns ['y_true', 'y_pred'], indexed by timestamp.
    threshold : float
        Signal threshold in log-return units.
    cost_per_trade : float
        One-way cost as a fraction of notional.
    slippage : float
        Additional one-way slippage fraction.
    annualise_factor : int
        Number of bars per year for Sharpe annualisation.

    Returns
    -------
    pd.DataFrame
        Original oos with additional columns:
        position, tc, strategy_ret, cumulative_ret.
    """
    bt = oos[["y_true", "y_pred"]].copy()

    # Signal → position
    bt["position"] = 0
    bt.loc[bt["y_pred"] >  threshold,  "position"] = 1
    bt.loc[bt["y_pred"] < -threshold,  "position"] = -1

    # Transaction costs: charged on CHANGE in position
    pos_change = bt["position"].diff().abs()
    pos_change.iloc[0] = abs(bt["position"].iloc[0])  # opening cost on first bar
    bt["tc"] = pos_change * (cost_per_trade + slippage)

    # Strategy return per bar
    bt["strategy_ret"] = bt["position"] * bt["y_true"] - bt["tc"]

    # Cumulative log return
    bt["cumulative_ret"] = bt["strategy_ret"].cumsum()

    return bt


def compute_backtest_stats(bt: pd.DataFrame, annualise_factor: int = 98_000) -> dict:
    """
    Summarise backtest performance.

    Parameters
    ----------
    bt : pd.DataFrame
        Output of run_backtest().
    annualise_factor : int
        Bars per year for Sharpe proxy.

    Returns
    -------
    dict of performance statistics.
    """
    rets = bt["strategy_ret"]
    positions = bt["position"]

    # Cumulative return (log, then convert to simple)
    total_log_ret = rets.sum()
    total_simple_ret = float(np.expm1(total_log_ret))

    # Sharpe (annualised) — using log returns directly
    mu = rets.mean()
    sigma = rets.std()
    sharpe = float((mu / sigma) * np.sqrt(annualise_factor)) if sigma > 0 else float("nan")

    # Max drawdown (log and simple)
    cum = bt["cumulative_ret"]
    running_max = cum.cummax()
    drawdown = cum - running_max
    max_dd = float(drawdown.min())
    max_dd_simple = float(np.expm1(max_dd))  # convert to simple %

    # Wipeout detection: flag if capital dropped 95%+ at any point
    wiped_out = max_dd_simple <= -0.95
    wipeout_bar = None
    if wiped_out:
        wipeout_bar = (np.expm1(cum) <= -0.95).idxmax()

    # Turnover: number of position changes / total bars
    trades = (positions.diff().abs() > 0).sum()
    total_bars = len(bt)
    turnover = float(trades / total_bars)

    # Hit rate: among bars where we had a non-zero position, fraction we were correct.
    # Uses y_pred if available (ML backtest), otherwise falls back to position sign.
    active = bt[positions != 0]
    if len(active):
        if "y_pred" in bt.columns:
            signal = np.sign(active["y_pred"])
        else:
            signal = active["position"]
        hit_rate = float((signal == np.sign(active["y_true"])).mean())
        avg_ret_per_trade = float(active["strategy_ret"].mean())
    else:
        hit_rate = float("nan")
        avg_ret_per_trade = float("nan")

    stats = {
        "total_log_return": float(total_log_ret),
        "total_simple_return": total_simple_ret,
        "annualised_sharpe": sharpe,
        "max_drawdown_log": max_dd,
        "max_drawdown_pct": max_dd_simple,
        "wiped_out": wiped_out,
        "wipeout_bar": wipeout_bar,
        "n_trades": int(trades),
        "n_bars": total_bars,
        "turnover_rate": turnover,
        "hit_rate": hit_rate,
        "avg_ret_per_trade": avg_ret_per_trade,
        "fraction_long": float((positions == 1).mean()),
        "fraction_short": float((positions == -1).mean()),
        "fraction_flat": float((positions == 0).mean()),
    }
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Position-based backtest (for strategies that output positions directly)
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest_from_positions(
    prices: pd.Series,
    positions: pd.Series,
    cost_per_trade: float = 0.0001,
    slippage: float = 0.00005,
    annualise_factor: int = 98_000,
) -> pd.DataFrame:
    """
    Run a backtest given a price series and a pre-computed position series.

    Parameters
    ----------
    prices : pd.Series
        Close prices (or any price series), indexed by timestamp.
    positions : pd.Series
        Position signal: +1 (long), -1 (short), 0 (flat). Same index as prices.
    cost_per_trade : float
        One-way cost as a fraction of notional.
    slippage : float
        Additional one-way slippage fraction.
    annualise_factor : int
        Number of bars per year for Sharpe annualisation.

    Returns
    -------
    pd.DataFrame with columns: y_true, position, tc, strategy_ret, cumulative_ret.
    """
    bt = pd.DataFrame(index=prices.index)
    bt["y_true"] = np.log(prices).diff().shift(-1)  # next-bar log return
    bt["position"] = positions.reindex(bt.index).fillna(0).astype(int)

    # Drop last row — no y_true available for it
    bt = bt.iloc[:-1].copy()

    pos_change = bt["position"].diff().abs()
    pos_change.iloc[0] = abs(bt["position"].iloc[0])
    bt["tc"] = pos_change * (cost_per_trade + slippage)

    bt["strategy_ret"] = bt["position"] * bt["y_true"] - bt["tc"]
    bt["cumulative_ret"] = bt["strategy_ret"].cumsum()

    return bt


# ─────────────────────────────────────────────────────────────────────────────
# Threshold sweep
# ─────────────────────────────────────────────────────────────────────────────

def threshold_sweep(
    oos: pd.DataFrame,
    thresholds: list[float],
    cost_per_trade: float = 0.0001,
    slippage: float = 0.00005,
    annualise_factor: int = 98_000,
) -> pd.DataFrame:
    """
    Run the backtest for each threshold and return a summary table.

    Returns
    -------
    pd.DataFrame indexed by threshold.
    """
    rows = []
    for thr in thresholds:
        bt = run_backtest(oos, threshold=thr, cost_per_trade=cost_per_trade,
                          slippage=slippage, annualise_factor=annualise_factor)
        stats = compute_backtest_stats(bt, annualise_factor=annualise_factor)
        stats["threshold"] = thr
        rows.append(stats)

    sweep_df = pd.DataFrame(rows).set_index("threshold")

    # Log summary
    logger.info("\n=== Threshold Sweep ===\n%s\n",
                sweep_df[["annualised_sharpe", "total_log_return",
                           "max_drawdown", "hit_rate", "n_trades"]].to_string())
    return sweep_df
