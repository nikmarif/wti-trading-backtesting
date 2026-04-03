#!/usr/bin/env python3
"""
run_strategy_backtest.py — standalone, strategy-agnostic backtester.

Usage
-----
    # List available strategies
    python scripts/run_strategy_backtest.py --list

    # Run with defaults
    python scripts/run_strategy_backtest.py --strategy ma_crossover
    python scripts/run_strategy_backtest.py --strategy rsi
    python scripts/run_strategy_backtest.py --strategy bollinger

    # Override strategy parameters
    python scripts/run_strategy_backtest.py --strategy ma_crossover --params fast=3 slow=15
    python scripts/run_strategy_backtest.py --strategy rsi --params period=7 oversold=25 overbought=75

    # Custom data / cost assumptions
    python scripts/run_strategy_backtest.py --strategy bollinger --config path/to.yaml \\
        --cost 0.0002 --slippage 0.0001

Notes
-----
- Data is loaded via the same config/data_loader as the ML pipeline.
- Positions are generated without lookahead: position[t] uses only data[0..t].
- Transaction costs follow the same model as the ML backtest.
- To add a new strategy, subclass BaseStrategy and register it in
  src/strategies/technical.py REGISTRY.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.backtest import compute_backtest_stats, run_backtest_from_positions
from src.data_loader import load_and_clean
from src.plotting import plot_strategy_trades
from src.strategies.technical import REGISTRY, build_strategy, list_strategies
from src.utils import ensure_dir, get_logger, load_config

logger = get_logger("strategy_backtest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strategy-agnostic backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available strategies and exit",
    )
    parser.add_argument(
        "--strategy", default="ma_crossover",
        help="Strategy name (see --list for options)",
    )
    parser.add_argument(
        "--params", nargs="*", metavar="key=value",
        help="Strategy parameter overrides, e.g. fast=3 slow=15",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to config YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--cost", type=float, default=None,
        help="Cost per trade fraction (overrides config)",
    )
    parser.add_argument(
        "--slippage", type=float, default=None,
        help="Slippage fraction (overrides config)",
    )
    parser.add_argument(
        "--annualise-factor", type=int, default=None,
        dest="annualise_factor",
        help="Bars per year for Sharpe (overrides config)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save a trade chart PNG to the artifacts directory",
    )
    parser.add_argument(
        "--plot-bars", type=int, default=500, dest="plot_bars",
        help="Number of bars to show in the trade chart (default: 500)",
    )
    return parser.parse_args()


def parse_overrides(params: list[str] | None) -> dict:
    """Convert ['fast=3', 'slow=15'] → {'fast': 3, 'slow': 15} (auto-typed)."""
    if not params:
        return {}
    overrides = {}
    for item in params:
        if "=" not in item:
            raise ValueError(f"Invalid param override '{item}'. Expected key=value.")
        k, v = item.split("=", 1)
        # Try int, then float, then str
        for cast in (int, float, str):
            try:
                overrides[k.strip()] = cast(v.strip())
                break
            except ValueError:
                continue
    return overrides


def main() -> None:
    args = parse_args()

    if args.list:
        print("\nAvailable strategies:")
        for name in list_strategies():
            cls, defaults = REGISTRY[name]
            print(f"  {name:<20} {cls.__name__}  defaults={defaults}")
        print()
        return

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    artifacts_dir = ensure_dir(cfg["output"]["artifacts_dir"])
    bt_cfg = cfg.get("backtest", {})

    cost_per_trade = args.cost if args.cost is not None else bt_cfg.get("cost_per_trade", 0.0001)
    slippage = args.slippage if args.slippage is not None else bt_cfg.get("slippage", 0.00005)
    annualise_factor = (
        args.annualise_factor if args.annualise_factor is not None
        else bt_cfg.get("annualise_factor", 98_000)
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Loading data...")
    ohlcv = load_and_clean(cfg)
    logger.info("Loaded %d bars (%s → %s)", len(ohlcv), ohlcv.index.min(), ohlcv.index.max())

    # ── Strategy ──────────────────────────────────────────────────────────────
    overrides = parse_overrides(args.params)
    strategy = build_strategy(args.strategy, overrides)
    logger.info("Strategy: %s", strategy)

    logger.info("Generating positions...")
    positions = strategy.generate_positions(ohlcv)
    n_long = (positions == 1).sum()
    n_short = (positions == -1).sum()
    n_flat = (positions == 0).sum()
    logger.info("Positions — long: %d | short: %d | flat: %d", n_long, n_short, n_flat)

    # ── Backtest ──────────────────────────────────────────────────────────────
    logger.info("Running backtest...")
    bt = run_backtest_from_positions(
        prices=ohlcv["close"],
        positions=positions,
        cost_per_trade=cost_per_trade,
        slippage=slippage,
        annualise_factor=annualise_factor,
    )
    stats = compute_backtest_stats(bt, annualise_factor=annualise_factor)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  STRATEGY BACKTEST — {args.strategy.upper()}")
    print("=" * 60)
    print(f"  Strategy:          {strategy}")
    print(f"  Bars:              {len(ohlcv):>10,}")
    print(f"  Backtest bars:     {stats['n_bars']:>10,}")
    print(f"  Cost per trade:    {cost_per_trade:>10.5f}")
    print(f"  Slippage:          {slippage:>10.5f}")
    print()
    print("  Performance")
    print(f"    Sharpe (ann.):   {stats['annualised_sharpe']:>12.3f}")
    print(f"    Total log ret:   {stats['total_log_return']:>12.4f}")
    print(f"    Total simple ret:{stats['total_simple_return']:>12.4f}")
    print(f"    Max drawdown:    {stats['max_drawdown']:>12.4f}")
    print(f"    Hit rate:        {stats['hit_rate']:>12.4f}")
    print(f"    N trades:        {stats['n_trades']:>12,}")
    print(f"    Turnover rate:   {stats['turnover_rate']:>12.4f}")
    print(f"    Avg ret/trade:   {stats['avg_ret_per_trade']:>12.6f}")
    print()
    print("  Position breakdown")
    print(f"    Long:            {stats['fraction_long']:>12.2%}")
    print(f"    Short:           {stats['fraction_short']:>12.2%}")
    print(f"    Flat:            {stats['fraction_flat']:>12.2%}")
    print("=" * 60 + "\n")

    # ── Trade chart ───────────────────────────────────────────────────────────
    if args.plot:
        chart_path = plot_strategy_trades(
            bt=bt,
            strategy_name=str(strategy),
            artifacts_dir=artifacts_dir,
            max_bars=args.plot_bars,
        )
        print(f"  Chart saved → {chart_path}\n")


if __name__ == "__main__":
    main()
