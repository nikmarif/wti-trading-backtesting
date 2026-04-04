"""
base.py — Abstract base class for all backtest strategies.

A strategy takes OHLCV data and produces a position series: +1, -1, or 0.
No lookahead: positions at bar t must be based only on data up to bar t.

To add a new strategy:
  1. Subclass BaseStrategy.
  2. Implement generate_positions().
  3. Register it in the REGISTRY dict in technical.py (or your own module).
"""

from __future__ import annotations

import abc

import pandas as pd


class BaseStrategy(abc.ABC):
    """
    Abstract strategy interface.

    Subclasses must implement generate_positions(), which accepts an OHLCV
    DataFrame (columns: open, high, low, close, volume; DatetimeIndex) and
    returns a pd.Series of integer positions aligned to the same index.
    """

    @abc.abstractmethod
    def generate_positions(self, ohlcv: pd.DataFrame) -> pd.Series:
        """
        Compute positions for each bar.

        Parameters
        ----------
        ohlcv : pd.DataFrame
            OHLCV data with a DatetimeIndex. Columns: open, high, low, close, volume.

        Returns
        -------
        pd.Series[int]
            +1 = long, -1 = short, 0 = flat. Same index as ohlcv.
        """

    def get_indicators(self, ohlcv: pd.DataFrame) -> dict[str, pd.Series]:
        """
        Optional: return indicator series to overlay on the trade chart.
        Override in subclasses to expose strategy-specific lines (MAs, bands, etc.).
        Returns an empty dict by default.
        """
        return {}

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({params})"
