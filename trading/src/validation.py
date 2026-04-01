"""
validation.py — time-series walk-forward cross-validation.

Two strategies are provided:
  - ExpandingWindowCV  (default, recommended): train grows with each fold.
  - RollingWindowCV:   fixed-size training window slides forward.

Both yield (train_idx, val_idx) integer index tuples, never touching
future validation data during training.

Key invariant: for every fold, max(train_idx) < min(val_idx).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Tuple

import numpy as np
import pandas as pd

from .utils import get_logger

logger = get_logger(__name__)

SplitGen = Generator[Tuple[np.ndarray, np.ndarray], None, None]


# ─────────────────────────────────────────────────────────────────────────────
# Expanding window (recommended)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExpandingWindowCV:
    """
    Walk-forward cross-validation with an expanding training window.

    Parameters
    ----------
    min_train_bars : int
        Minimum number of training bars in the first fold.
    val_bars : int
        Number of bars in each validation window.
    step_bars : int
        How many bars to advance the validation window between folds.
        A value smaller than val_bars creates overlapping val windows;
        equal to val_bars gives non-overlapping (recommended).
    """
    min_train_bars: int = 15_000
    val_bars: int = 5_000
    step_bars: int = 2_500

    def split(self, X: pd.DataFrame) -> SplitGen:
        """
        Yield (train_indices, val_indices) for each fold.

        Indices are integer positional (suitable for .iloc[]).
        """
        n = len(X)
        train_end = self.min_train_bars

        fold = 0
        while train_end + self.val_bars <= n:
            train_idx = np.arange(0, train_end)
            val_end = min(train_end + self.val_bars, n)
            val_idx = np.arange(train_end, val_end)

            logger.info(
                "Fold %d | train [0, %d) = %d bars | val [%d, %d) = %d bars",
                fold, train_end, len(train_idx),
                train_end, val_end, len(val_idx),
            )
            yield train_idx, val_idx

            train_end += self.step_bars
            fold += 1

        if fold == 0:
            raise ValueError(
                f"Not enough data for a single fold. "
                f"Need at least {self.min_train_bars + self.val_bars} bars, "
                f"got {n}. Reduce min_train_bars or val_bars in config."
            )
        logger.info("Walk-forward complete: %d folds.", fold)

    def n_splits(self, X: pd.DataFrame) -> int:
        """Return the total number of folds without materialising them."""
        n = len(X)
        count = 0
        train_end = self.min_train_bars
        while train_end + self.val_bars <= n:
            count += 1
            train_end += self.step_bars
        return count


# ─────────────────────────────────────────────────────────────────────────────
# Rolling (fixed-size) window — optional alternative
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RollingWindowCV:
    """
    Walk-forward cross-validation with a fixed-size sliding training window.

    Use when you believe the data-generating process is non-stationary and
    old data hurts more than it helps.

    Parameters
    ----------
    train_bars : int
        Fixed number of training bars.
    val_bars : int
        Number of bars in each validation window.
    step_bars : int
        Bars to advance per fold.
    """
    train_bars: int = 20_000
    val_bars: int = 5_000
    step_bars: int = 2_500

    def split(self, X: pd.DataFrame) -> SplitGen:
        n = len(X)
        start = 0
        fold = 0

        while start + self.train_bars + self.val_bars <= n:
            train_idx = np.arange(start, start + self.train_bars)
            val_start = start + self.train_bars
            val_end = min(val_start + self.val_bars, n)
            val_idx = np.arange(val_start, val_end)

            logger.info(
                "Fold %d | train [%d, %d) = %d bars | val [%d, %d) = %d bars",
                fold, start, start + self.train_bars, len(train_idx),
                val_start, val_end, len(val_idx),
            )
            yield train_idx, val_idx

            start += self.step_bars
            fold += 1

        if fold == 0:
            raise ValueError(
                f"Not enough data. Need {self.train_bars + self.val_bars} bars, got {n}."
            )
        logger.info("Walk-forward complete: %d folds.", fold)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def make_cv(val_cfg: dict, strategy: str = "expanding"):
    """
    Construct a CV splitter from the 'validation' config section.

    Parameters
    ----------
    val_cfg : dict
        The 'validation' section of config.yaml.
    strategy : str
        'expanding' (default) or 'rolling'.
    """
    if strategy == "expanding":
        return ExpandingWindowCV(
            min_train_bars=val_cfg["min_train_bars"],
            val_bars=val_cfg["val_bars"],
            step_bars=val_cfg["step_bars"],
        )
    elif strategy == "rolling":
        return RollingWindowCV(
            train_bars=val_cfg.get("rolling_train_bars", val_cfg["min_train_bars"]),
            val_bars=val_cfg["val_bars"],
            step_bars=val_cfg["step_bars"],
        )
    else:
        raise ValueError(f"Unknown CV strategy: {strategy!r}. Use 'expanding' or 'rolling'.")
