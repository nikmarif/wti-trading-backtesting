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
    Moving-average crossover with fast-MA touch exit.

    Entry
    -----
    - fast MA crosses above slow MA → go long  (+1)
    - fast MA crosses below slow MA → go short (-1)

    Exit
    ----
    - Long:  price touches or crosses below the fast MA → go flat (0)
    - Short: price touches or crosses above the fast MA → go flat (0)

    A new trade only opens at the next crossover after going flat.

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

        close_arr = close.values
        fast_arr = fast_ma.values
        slow_arr = slow_ma.values
        n = len(close_arr)
        pos = np.zeros(n, dtype=int)

        current = 0  # current position: 0, 1, or -1
        for i in range(1, n):
            if np.isnan(fast_arr[i]) or np.isnan(slow_arr[i]):
                pos[i] = 0
                current = 0
                continue

            if current == 0:
                # Look for a crossover to enter
                crossed_up = fast_arr[i] > slow_arr[i] and fast_arr[i - 1] <= slow_arr[i - 1]
                crossed_dn = fast_arr[i] < slow_arr[i] and fast_arr[i - 1] >= slow_arr[i - 1]
                if crossed_up:
                    current = 1
                elif crossed_dn:
                    current = -1

            elif current == 1:
                # Exit long if price touches or drops below the fast MA
                if close_arr[i] <= fast_arr[i]:
                    current = 0

            elif current == -1:
                # Exit short if price touches or rises above the fast MA
                if close_arr[i] >= fast_arr[i]:
                    current = 0

            pos[i] = current

        return pd.Series(pos, index=ohlcv.index, dtype=int)

    def get_indicators(self, ohlcv: pd.DataFrame) -> dict[str, pd.Series]:
        close = ohlcv["close"]
        if self.ma_type == "ema":
            fast_ma = close.ewm(span=self.fast, adjust=False).mean()
            slow_ma = close.ewm(span=self.slow, adjust=False).mean()
        else:
            fast_ma = close.rolling(self.fast).mean()
            slow_ma = close.rolling(self.slow).mean()
        return {
            f"Fast MA ({self.fast})": fast_ma,
            f"Slow MA ({self.slow})": slow_ma,
        }


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
# Fixed TP/SL Strategy
# ─────────────────────────────────────────────────────────────────────────────

class FixedTPSLStrategy(BaseStrategy):
    """
    MA crossover entry with fixed take-profit and stop-loss exit.

    Simulates holding a leveraged position with fixed capital:
      - Notional = capital * leverage
      - TP fires when price moves  +(take_profit / leverage) from entry price
      - SL fires when price moves  -(stop_loss  / leverage) from entry price

    Example with defaults (capital=800, leverage=5, tp=0.20, sl=0.10):
      - Notional = $4,000
      - TP at +4% price move  (+20% on $800)
      - SL at -2% price move  (-10% on $800)

    Entry direction is determined by MA crossover (fast/slow).
    After a TP or SL exit, the strategy waits for the next crossover to re-enter.

    Parameters
    ----------
    capital : float
        Starting capital in dollars.
    leverage : float
        Leverage multiplier.
    take_profit : float
        Target profit as a fraction of capital (e.g. 0.20 = 20%).
    stop_loss : float
        Maximum loss as a fraction of capital (e.g. 0.10 = 10%).
    fast : int
        Fast MA window for entry signal.
    slow : int
        Slow MA window for entry signal.
    ma_type : str
        'sma' or 'ema'.
    """

    def __init__(
        self,
        capital: float = 800.0,
        leverage: float = 5.0,
        take_profit: float = 0.20,
        stop_loss: float = 0.10,
        fast: int = 5,
        slow: int = 20,
        ma_type: str = "sma",
    ):
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be < slow ({slow})")
        self.capital = capital
        self.leverage = leverage
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.fast = fast
        self.slow = slow
        self.ma_type = ma_type.lower()

        # Price move thresholds on the underlying (before leverage)
        self._tp_move = take_profit / leverage   # e.g. 0.04
        self._sl_move = stop_loss  / leverage    # e.g. 0.02

    def generate_positions(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"]
        if self.ma_type == "ema":
            fast_ma = close.ewm(span=self.fast, adjust=False).mean()
            slow_ma = close.ewm(span=self.slow, adjust=False).mean()
        else:
            fast_ma = close.rolling(self.fast).mean()
            slow_ma = close.rolling(self.slow).mean()

        close_arr = close.values
        fast_arr  = fast_ma.values
        slow_arr  = slow_ma.values
        n = len(close_arr)
        pos = np.zeros(n, dtype=int)

        current     = 0      # current position: 0, 1, or -1
        entry_price = None

        for i in range(1, n):
            if np.isnan(fast_arr[i]) or np.isnan(slow_arr[i]):
                pos[i] = 0
                current = 0
                continue

            if current == 0:
                # Wait for a crossover to enter
                crossed_up = fast_arr[i] > slow_arr[i] and fast_arr[i-1] <= slow_arr[i-1]
                crossed_dn = fast_arr[i] < slow_arr[i] and fast_arr[i-1] >= slow_arr[i-1]
                if crossed_up:
                    current = 1
                    entry_price = close_arr[i]
                elif crossed_dn:
                    current = -1
                    entry_price = close_arr[i]

            elif current == 1:
                move = (close_arr[i] - entry_price) / entry_price
                if move >= self._tp_move or move <= -self._sl_move:
                    current = 0
                    entry_price = None

            elif current == -1:
                move = (entry_price - close_arr[i]) / entry_price
                if move >= self._tp_move or move <= -self._sl_move:
                    current = 0
                    entry_price = None

            pos[i] = current

        return pd.Series(pos, index=ohlcv.index, dtype=int)

    def get_indicators(self, ohlcv: pd.DataFrame) -> dict[str, pd.Series]:
        close = ohlcv["close"]
        if self.ma_type == "ema":
            fast_ma = close.ewm(span=self.fast, adjust=False).mean()
            slow_ma = close.ewm(span=self.slow, adjust=False).mean()
        else:
            fast_ma = close.rolling(self.fast).mean()
            slow_ma = close.rolling(self.slow).mean()
        return {
            f"Fast MA ({self.fast})": fast_ma,
            f"Slow MA ({self.slow})": slow_ma,
        }


# ─────────────────────────────────────────────────────────────────────────────
# MA Second Touch Strategy
# ─────────────────────────────────────────────────────────────────────────────

class MASecondTouchStrategy(BaseStrategy):
    """
    MA crossover entry, second fast-MA touch exit, with % stop loss.

    Entry
    -----
    - Fast MA crosses above slow MA → go long
    - Fast MA crosses below slow MA → go short

    Exit
    ----
    - Count how many times price touches the fast MA during the trade.
      A "touch" is when price crosses from the trade side to the other side of
      the fast MA (consecutive bars on the same side count as one touch).
    - First touch: ignore — could just be a pullback.
    - Second touch: exit — the move is likely exhausted.
    - Stop loss: if price moves stop_loss% against entry at any bar → exit immediately.

    Parameters
    ----------
    stop_loss : float
        Max price move against entry before stopping out (e.g. 0.02 = 2%).
    fast : int
        Fast MA window.
    slow : int
        Slow MA window.
    ma_type : str
        'sma' or 'ema'.
    """

    def __init__(
        self,
        stop_loss: float = 0.02,
        fast: int = 5,
        slow: int = 20,
        ma_type: str = "sma",
    ):
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be < slow ({slow})")
        self.stop_loss = stop_loss
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

        close_arr = close.values
        fast_arr  = fast_ma.values
        slow_arr  = slow_ma.values
        n = len(close_arr)
        pos = np.zeros(n, dtype=int)

        current      = 0      # current position: 0, 1, or -1
        entry_price  = None
        touch_count  = 0
        in_touch     = False  # whether price is currently on the MA side

        for i in range(1, n):
            if np.isnan(fast_arr[i]) or np.isnan(slow_arr[i]):
                pos[i] = 0
                current = 0
                continue

            if current == 0:
                # Wait for crossover to enter
                crossed_up = fast_arr[i] > slow_arr[i] and fast_arr[i-1] <= slow_arr[i-1]
                crossed_dn = fast_arr[i] < slow_arr[i] and fast_arr[i-1] >= slow_arr[i-1]
                if crossed_up:
                    current = 1
                    entry_price = close_arr[i]
                    touch_count = 0
                    in_touch = False
                elif crossed_dn:
                    current = -1
                    entry_price = close_arr[i]
                    touch_count = 0
                    in_touch = False

            elif current == 1:
                # Stop loss: price drops stop_loss% below entry
                if close_arr[i] <= entry_price * (1 - self.stop_loss):
                    current = 0
                    entry_price = None
                    touch_count = 0
                    in_touch = False
                else:
                    # Count touches: price at or below fast MA = touching for a long
                    touching = close_arr[i] <= fast_arr[i]
                    if touching and not in_touch:
                        touch_count += 1
                        in_touch = True
                    elif not touching:
                        in_touch = False
                    if touch_count >= 2:
                        current = 0
                        entry_price = None
                        touch_count = 0
                        in_touch = False

            elif current == -1:
                # Stop loss: price rises stop_loss% above entry
                if close_arr[i] >= entry_price * (1 + self.stop_loss):
                    current = 0
                    entry_price = None
                    touch_count = 0
                    in_touch = False
                else:
                    # Count touches: price at or above fast MA = touching for a short
                    touching = close_arr[i] >= fast_arr[i]
                    if touching and not in_touch:
                        touch_count += 1
                        in_touch = True
                    elif not touching:
                        in_touch = False
                    if touch_count >= 2:
                        current = 0
                        entry_price = None
                        touch_count = 0
                        in_touch = False

            pos[i] = current

        return pd.Series(pos, index=ohlcv.index, dtype=int)

    def get_indicators(self, ohlcv: pd.DataFrame) -> dict[str, pd.Series]:
        close = ohlcv["close"]
        if self.ma_type == "ema":
            fast_ma = close.ewm(span=self.fast, adjust=False).mean()
            slow_ma = close.ewm(span=self.slow, adjust=False).mean()
        else:
            fast_ma = close.rolling(self.fast).mean()
            slow_ma = close.rolling(self.slow).mean()
        return {
            f"Fast MA ({self.fast})": fast_ma,
            f"Slow MA ({self.slow})": slow_ma,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Registry — maps CLI name → (class, default params)
# Add your new strategies here so the CLI can discover them.
# ─────────────────────────────────────────────────────────────────────────────

REGISTRY: dict[str, tuple[type[BaseStrategy], dict]] = {
    "ma_crossover": (MACrossoverStrategy, {"fast": 5, "slow": 20, "ma_type": "sma"}),
    "ema_crossover": (MACrossoverStrategy, {"fast": 5, "slow": 20, "ma_type": "ema"}),
    "rsi": (RSIStrategy, {"period": 14, "oversold": 30, "overbought": 70}),
    "bollinger": (BollingerBandStrategy, {"window": 20, "n_std": 2.0}),
    "fixed_tpsl": (FixedTPSLStrategy, {
        "capital": 800.0, "leverage": 5.0,
        "take_profit": 0.20, "stop_loss": 0.10,
        "fast": 5, "slow": 20, "ma_type": "sma",
    }),
    "ma_second_touch": (MASecondTouchStrategy, {
        "stop_loss": 0.02, "fast": 5, "slow": 20, "ma_type": "sma",
    }),
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
