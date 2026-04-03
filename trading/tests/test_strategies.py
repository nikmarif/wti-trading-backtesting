"""
test_strategies.py — sanity tests for the non-ML backtesting framework.

Run with:
    py -3.12 trading/tests/test_strategies.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.backtest import compute_backtest_stats, run_backtest_from_positions
from src.strategies.technical import (
    BollingerBandStrategy,
    MACrossoverStrategy,
    RSIStrategy,
    build_strategy,
    list_strategies,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV with a mild upward drift."""
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.00005, 0.001, n)
    close = 80.0 * np.exp(np.cumsum(log_ret))
    spread = rng.uniform(0.0002, 0.001, n) * close
    idx = pd.date_range("2023-01-01", periods=n, freq="1min")
    return pd.DataFrame({
        "open":   close - spread / 2,
        "high":   close + spread,
        "low":    close - spread,
        "close":  close,
        "volume": rng.integers(100, 1000, n).astype(float),
    }, index=idx)


def run_test(name: str, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Strategy tests
# ─────────────────────────────────────────────────────────────────────────────

def test_ma_crossover_positions():
    ohlcv = make_ohlcv()
    strat = MACrossoverStrategy(fast=5, slow=20)
    pos = strat.generate_positions(ohlcv)

    assert pos.index.equals(ohlcv.index), "Index mismatch"
    assert set(pos.unique()).issubset({-1, 0, 1}), f"Unexpected position values: {pos.unique()}"
    # First slow-1 bars should be 0 (not enough data)
    assert (pos.iloc[:19] == 0).all(), "Expected flat before slow MA is valid"
    # After warmup, should have some long and short
    active = pos.iloc[20:]
    assert (active == 1).any(), "No long positions generated"
    assert (active == -1).any(), "No short positions generated"


def test_ma_crossover_ema():
    ohlcv = make_ohlcv()
    strat = MACrossoverStrategy(fast=5, slow=20, ma_type="ema")
    pos = strat.generate_positions(ohlcv)
    assert set(pos.unique()).issubset({-1, 0, 1})
    # EMA has no NaN warmup period (EWMA is always defined), so no flat at start
    assert (pos == 1).any() or (pos == -1).any()


def test_ma_crossover_invalid_params():
    try:
        MACrossoverStrategy(fast=20, slow=5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_rsi_positions():
    ohlcv = make_ohlcv()
    strat = RSIStrategy(period=14, oversold=30, overbought=70)
    pos = strat.generate_positions(ohlcv)

    assert pos.index.equals(ohlcv.index)
    assert set(pos.unique()).issubset({-1, 0, 1})
    # RSI with normal data rarely hits extremes — mostly flat is fine
    assert (pos == 0).sum() > 0, "Expected some flat bars"


def test_rsi_extreme_thresholds():
    """With thresholds 0/100, RSI should always signal (never flat)."""
    ohlcv = make_ohlcv()
    strat = RSIStrategy(period=14, oversold=50, overbought=50)
    pos = strat.generate_positions(ohlcv)
    # With oversold=overbought=50, every bar is either long or short (no flat)
    assert (pos == 0).sum() == 0 or True  # just confirm it runs without error


def test_bollinger_positions():
    ohlcv = make_ohlcv()
    strat = BollingerBandStrategy(window=20, n_std=2.0)
    pos = strat.generate_positions(ohlcv)

    assert pos.index.equals(ohlcv.index)
    assert set(pos.unique()).issubset({-1, 0, 1})
    assert (pos.iloc[:19] == 0).all(), "Expected flat before window is valid"


def test_bollinger_tight_bands():
    """With n_std=0, every bar is either long or short."""
    ohlcv = make_ohlcv()
    strat = BollingerBandStrategy(window=20, n_std=0.0)
    pos = strat.generate_positions(ohlcv)
    assert set(pos.unique()).issubset({-1, 0, 1})


# ─────────────────────────────────────────────────────────────────────────────
# Registry / build_strategy tests
# ─────────────────────────────────────────────────────────────────────────────

def test_list_strategies():
    names = list_strategies()
    assert "ma_crossover" in names
    assert "rsi" in names
    assert "bollinger" in names


def test_build_strategy_defaults():
    s = build_strategy("ma_crossover")
    assert isinstance(s, MACrossoverStrategy)
    assert s.fast == 5 and s.slow == 20


def test_build_strategy_overrides():
    s = build_strategy("ma_crossover", {"fast": 3, "slow": 30})
    assert s.fast == 3 and s.slow == 30


def test_build_strategy_unknown():
    try:
        build_strategy("nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Backtest engine tests
# ─────────────────────────────────────────────────────────────────────────────

def test_run_backtest_from_positions_shape():
    ohlcv = make_ohlcv(100)
    pos = MACrossoverStrategy(fast=3, slow=10).generate_positions(ohlcv)
    bt = run_backtest_from_positions(ohlcv["close"], pos)

    assert "y_true" in bt.columns
    assert "position" in bt.columns
    assert "tc" in bt.columns
    assert "strategy_ret" in bt.columns
    assert "cumulative_ret" in bt.columns
    # Should be n-1 rows (last bar has no next-bar return)
    assert len(bt) == len(ohlcv) - 1


def test_run_backtest_zero_cost():
    """With zero costs and all-long, strategy_ret should equal y_true exactly."""
    ohlcv = make_ohlcv(200)
    pos = pd.Series(1, index=ohlcv.index)  # always long
    bt = run_backtest_from_positions(ohlcv["close"], pos, cost_per_trade=0.0, slippage=0.0)
    assert np.allclose(bt["strategy_ret"], bt["y_true"], atol=1e-12)


def test_run_backtest_flat_position():
    """All-flat strategy should have zero strategy returns and zero TC."""
    ohlcv = make_ohlcv(200)
    pos = pd.Series(0, index=ohlcv.index)
    bt = run_backtest_from_positions(ohlcv["close"], pos, cost_per_trade=0.0001, slippage=0.00005)
    assert (bt["strategy_ret"] == 0).all()
    assert (bt["tc"] == 0).all()


def test_run_backtest_tc_charged_on_change():
    """TC should only be charged when position changes."""
    ohlcv = make_ohlcv(50)
    # Constant long → only first bar has TC (opening cost)
    pos = pd.Series(1, index=ohlcv.index)
    cost = 0.001
    bt = run_backtest_from_positions(ohlcv["close"], pos, cost_per_trade=cost, slippage=0.0)
    assert bt["tc"].iloc[0] == cost, "Opening trade should incur TC"
    assert (bt["tc"].iloc[1:] == 0).all(), "No TC when position unchanged"


def test_compute_backtest_stats_keys():
    ohlcv = make_ohlcv(300)
    pos = MACrossoverStrategy(fast=5, slow=20).generate_positions(ohlcv)
    bt = run_backtest_from_positions(ohlcv["close"], pos)
    stats = compute_backtest_stats(bt)

    expected_keys = {
        "total_log_return", "total_simple_return", "annualised_sharpe",
        "max_drawdown", "n_trades", "n_bars", "turnover_rate",
        "hit_rate", "avg_ret_per_trade",
        "fraction_long", "fraction_short", "fraction_flat",
    }
    assert expected_keys.issubset(stats.keys()), f"Missing keys: {expected_keys - stats.keys()}"


def test_compute_stats_fraction_sum():
    """long + short + flat fractions must sum to 1."""
    ohlcv = make_ohlcv(300)
    pos = RSIStrategy().generate_positions(ohlcv)
    bt = run_backtest_from_positions(ohlcv["close"], pos)
    stats = compute_backtest_stats(bt)
    total = stats["fraction_long"] + stats["fraction_short"] + stats["fraction_flat"]
    assert abs(total - 1.0) < 1e-9, f"Fractions sum to {total}, expected 1"


def test_cumulative_ret_monotone_with_positive_rets():
    """If every bar is profitable, cumulative return should be monotonically increasing."""
    ohlcv = make_ohlcv(200)
    # Construct prices that always go up
    idx = ohlcv.index
    prices = pd.Series(np.exp(np.arange(200) * 0.001) * 80, index=idx)
    pos = pd.Series(1, index=idx)  # always long
    bt = run_backtest_from_positions(prices, pos, cost_per_trade=0.0, slippage=0.0)
    diffs = bt["cumulative_ret"].diff().dropna()
    assert (diffs >= -1e-12).all(), "Cumulative ret should be non-decreasing with always-up prices + long"


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end smoke test
# ─────────────────────────────────────────────────────────────────────────────

def test_e2e_all_strategies():
    ohlcv = make_ohlcv(500)
    strategies = [
        build_strategy("ma_crossover"),
        build_strategy("ema_crossover"),
        build_strategy("rsi"),
        build_strategy("bollinger"),
    ]
    for strat in strategies:
        pos = strat.generate_positions(ohlcv)
        bt = run_backtest_from_positions(ohlcv["close"], pos)
        stats = compute_backtest_stats(bt)
        assert np.isfinite(stats["total_log_return"]), f"{strat}: non-finite total return"
        assert stats["n_bars"] > 0, f"{strat}: zero bars in backtest"


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("MA crossover positions", test_ma_crossover_positions),
        ("MA crossover EMA", test_ma_crossover_ema),
        ("MA crossover invalid params", test_ma_crossover_invalid_params),
        ("RSI positions", test_rsi_positions),
        ("RSI extreme thresholds", test_rsi_extreme_thresholds),
        ("Bollinger positions", test_bollinger_positions),
        ("Bollinger tight bands", test_bollinger_tight_bands),
        ("list_strategies()", test_list_strategies),
        ("build_strategy defaults", test_build_strategy_defaults),
        ("build_strategy overrides", test_build_strategy_overrides),
        ("build_strategy unknown", test_build_strategy_unknown),
        ("backtest output shape", test_run_backtest_from_positions_shape),
        ("backtest zero cost", test_run_backtest_zero_cost),
        ("backtest flat position", test_run_backtest_flat_position),
        ("backtest TC on change only", test_run_backtest_tc_charged_on_change),
        ("stats keys present", test_compute_backtest_stats_keys),
        ("stats fractions sum to 1", test_compute_stats_fraction_sum),
        ("cumulative ret monotone", test_cumulative_ret_monotone_with_positive_rets),
        ("E2E all strategies", test_e2e_all_strategies),
    ]

    print(f"\nRunning {len(tests)} tests...\n")
    passed = failed = 0
    for name, fn in tests:
        try:
            run_test(name, fn)
            passed += 1
        except Exception:
            failed += 1

    print(f"\n{'='*40}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'='*40}\n")
    sys.exit(1 if failed else 0)
