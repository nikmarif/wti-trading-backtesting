"""
technical.py — Ready-to-use technical analysis strategies.

Available strategies
--------------------
  MACrossoverStrategy   : Fast/slow moving-average crossover
  RSIStrategy           : RSI mean-reversion (overbought/oversold)
  BollingerBandStrategy : Bollinger Band mean-reversion

Adding your own
---------------
  1. Subclass BaseStrategy and implement generate_positions().
  2. Add an entry to REGISTRY at the bottom of this file so the CLI can find it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseStrategy


# ─────────────────────────────────────────────────────────────────────────────
# MA Crossover
# ─────────────────────────────────────────────────────────────────────────────

class MACrossoverStrategy(BaseStrategy):
    """
    Simple moving-average crossover.

    Rules
    -----
    - fast_ma > slow_ma → long  (+1)
    - fast_ma < slow_ma → short (-1)
    - (optionally) flat until first valid slow_ma bar

    Parameters
    ----------
    fast : int
        Fast MA window (bars).
    slow : int
        Slow MA window (bars).
    ma_type : str
        'sma' (simple) or 'ema' (exponential).
    """

    def __init__(self, fast: int = 5, slow: int = 20, ma_type: str = "sma"):
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be < slow ({slow})")
        self.fast = fast
        self.slow = slow
        self.ma_type = ma_type.lower()

    def generate_positions(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"]
        if self.ma_type == "ema":
            fast_ma = close.ewm(span=self.fast, adjust=False).mean()
            slow_ma = close.ewm(span=self.slow, adjust=False).mean()
        else:
            fast_ma = close.rolling(self.fast).mean()
            slow_ma = close.rolling(self.slow).mean()

        position = pd.Series(0, index=ohlcv.index, dtype=int)
        valid = slow_ma.notna()
        position[valid & (fast_ma > slow_ma)] = 1
        position[valid & (fast_ma < slow_ma)] = -1
        return position


# ─────────────────────────────────────────────────────────────────────────────
# RSI Mean-Reversion
# ─────────────────────────────────────────────────────────────────────────────

class RSIStrategy(BaseStrategy):
    """
    RSI-based mean-reversion.

    Rules
    -----
    - RSI < oversold  → long  (+1)  (expect bounce up)
    - RSI > overbought → short (-1) (expect reversal down)
    - Otherwise        → flat  (0)

    Parameters
    ----------
    period : int
        RSI lookback window.
    oversold : float
        RSI level below which we go long (default 30).
    overbought : float
        RSI level above which we go short (default 70).
    """

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def _rsi(self, close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=self.period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=self.period - 1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def generate_positions(self, ohlcv: pd.DataFrame) -> pd.Series:
        rsi = self._rsi(ohlcv["close"])
        position = pd.Series(0, index=ohlcv.index, dtype=int)
        position[rsi < self.oversold] = 1
        position[rsi > self.overbought] = -1
        return position


# ─────────────────────────────────────────────────────────────────────────────
# Bollinger Band Mean-Reversion
# ─────────────────────────────────────────────────────────────────────────────

class BollingerBandStrategy(BaseStrategy):
    """
    Bollinger Band mean-reversion.

    Rules
    -----
    - price < lower band → long  (+1)  (expect reversion to mean)
    - price > upper band → short (-1)  (expect reversion to mean)
    - price between bands → flat (0)

    Parameters
    ----------
    window : int
        Rolling window for mean and std.
    n_std : float
        Number of standard deviations for the bands.
    """

    def __init__(self, window: int = 20, n_std: float = 2.0):
        self.window = window
        self.n_std = n_std

    def generate_positions(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"]
        mid = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()
        upper = mid + self.n_std * std
        lower = mid - self.n_std * std

        position = pd.Series(0, index=ohlcv.index, dtype=int)
        position[close < lower] = 1
        position[close > upper] = -1
        return position


# ─────────────────────────────────────────────────────────────────────────────
# Registry — maps CLI name → (class, default params)
# Add your new strategies here so the CLI can discover them.
# ─────────────────────────────────────────────────────────────────────────────

REGISTRY: dict[str, tuple[type[BaseStrategy], dict]] = {
    "ma_crossover": (MACrossoverStrategy, {"fast": 5, "slow": 20, "ma_type": "sma"}),
    "ema_crossover": (MACrossoverStrategy, {"fast": 5, "slow": 20, "ma_type": "ema"}),
    "rsi": (RSIStrategy, {"period": 14, "oversold": 30, "overbought": 70}),
    "bollinger": (BollingerBandStrategy, {"window": 20, "n_std": 2.0}),
}


def list_strategies() -> list[str]:
    return sorted(REGISTRY.keys())


def build_strategy(name: str, overrides: dict | None = None) -> BaseStrategy:
    """
    Instantiate a registered strategy by name, optionally overriding defaults.

    Parameters
    ----------
    name : str
        Key in REGISTRY.
    overrides : dict, optional
        Parameter overrides (e.g. {"fast": 3, "slow": 10}).

    Returns
    -------
    BaseStrategy instance.
    """
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list_strategies()}"
        )
    cls, defaults = REGISTRY[name]
    params = {**defaults, **(overrides or {})}
    return cls(**params)
