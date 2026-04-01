"""
feature_engineering.py — leakage-safe feature construction for 1-min oil returns.

LEAKAGE POLICY (read this before modifying)
============================================
Every feature at row t must be computable from information strictly available
at the END of bar t.  That means:

  - OHLCV values for bar t  ✓  (the bar has closed)
  - Log returns up to and including bar t  ✓
  - Rolling statistics using trailing windows (no center=True)  ✓
  - Any .shift(k) with k >= 1  ✓

FORBIDDEN:
  - .shift(-k) on any feature column   ✗  (looks into the future)
  - .rolling(w, center=True)           ✗  (uses future bars)
  - Any normalisation using statistics from future data  ✗

The TARGET uses .shift(-1) — this is intentional and isolated at the bottom
of this file.  Do not replicate that pattern anywhere else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import get_logger

logger = get_logger(__name__)

# Columns that are raw OHLCV — excluded from the final feature matrix
_OHLCV_COLS = {"open", "high", "low", "close", "volume"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log_returns(close: pd.Series, index: pd.DatetimeIndex,
                 max_gap_minutes: float = 5.0) -> pd.Series:
    """
    1-minute log return: ret[t] = log(close[t] / close[t-1]).

    Session-gap handling: if the time between bar t-1 and bar t exceeds
    max_gap_minutes (e.g. overnight, weekend), the return is set to NaN.
    These multi-hour returns are NOT 1-minute returns and would contaminate
    rolling statistics if left in.

    ret[0] is always NaN (no prior bar).
    """
    raw = np.log(close / close.shift(1))

    # Compute time gap between consecutive bars (in minutes)
    time_gap_minutes = index.to_series().diff().dt.total_seconds().div(60)

    # Null out returns that span a gap larger than max_gap_minutes
    session_open = time_gap_minutes > max_gap_minutes
    raw[session_open] = np.nan

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Feature groups
# ─────────────────────────────────────────────────────────────────────────────

def _lagged_returns(log_ret: pd.Series, max_lag: int = 20) -> pd.DataFrame:
    """
    A. Lagged return features: ret_lag_1 … ret_lag_{max_lag}.

    ret_lag_k at row t = log_ret[t - k].  All use only past bars.
    We shift by at least 1 so ret_lag_1 is already the PREVIOUS bar's return
    and there is no overlap with the current bar's return used elsewhere.
    """
    frames = {}
    for k in range(1, max_lag + 1):
        # LEAKAGE CHECK: shift(k) with k>=1 — safe
        frames[f"ret_lag_{k}"] = log_ret.shift(k)
    return pd.DataFrame(frames, index=log_ret.index)


def _rolling_return_stats(log_ret: pd.Series, windows: list[int]) -> pd.DataFrame:
    """
    B. Rolling mean and std of log returns over trailing windows.

    rolling(w) at row t uses bars [t-w+1 … t].  All past / current. Safe.
    min_periods=w enforces that we need a full window (avoids inflated early stats).
    """
    frames = {}
    for w in windows:
        roll = log_ret.rolling(w, min_periods=w)
        frames[f"ret_roll_mean_{w}"] = roll.mean()
        frames[f"ret_roll_std_{w}"] = roll.std()
    return pd.DataFrame(frames, index=log_ret.index)


def _momentum(log_ret: pd.Series, windows: list[int]) -> pd.DataFrame:
    """
    C. Cumulative log return (momentum) over trailing N bars.

    momentum_N[t] = sum(log_ret[t-N+1 … t]) = log(close[t] / close[t-N]).
    Uses rolling sum — no future data.
    """
    frames = {}
    for w in windows:
        # rolling sum of log returns over trailing w bars
        frames[f"momentum_{w}"] = log_ret.rolling(w, min_periods=w).sum()
    return pd.DataFrame(frames, index=log_ret.index)


def _candle_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    D. Candle structure features from bar t's OHLCV.

    All computed from the CURRENT completed bar — no future data.
    """
    o = df["open"]
    h = df["high"]
    l = df["low"]   # noqa: E741
    c = df["close"]

    hl_range = (h - l).replace(0.0, np.nan)
    body_high = np.maximum(o, c)
    body_low = np.minimum(o, c)

    feats = pd.DataFrame(index=df.index)
    feats["body"] = (c - o) / o.replace(0.0, np.nan)          # signed body
    feats["range"] = hl_range / o.replace(0.0, np.nan)        # total bar range
    feats["upper_wick"] = (h - body_high) / hl_range           # wick above body
    feats["lower_wick"] = (body_low - l) / hl_range            # wick below body
    # 0 = closed at low, 1 = closed at high
    feats["close_location"] = (c - l) / hl_range
    return feats


def _volume_features(volume: pd.Series, windows: list[int]) -> pd.DataFrame:
    """
    E. Volume level and spike ratios.

    All trailing rolling means — no future data.

    Zero-volume handling: HistData WTI (spot CFD) reports volume=0.
    When the rolling mean is zero, vol_spike is filled with 1.0 (neutral)
    rather than NaN so it doesn't silently kill all rows in dropna().
    """
    frames = {"volume_raw": volume}
    all_zero_volume = (volume == 0).all()
    for w in windows:
        roll_mean = volume.rolling(w, min_periods=w).mean()
        frames[f"vol_roll_mean_{w}"] = roll_mean
        if all_zero_volume:
            # No real volume data — fill spike with neutral value
            frames[f"vol_spike_{w}"] = 1.0
        else:
            frames[f"vol_spike_{w}"] = (
                volume / roll_mean.where(roll_mean > 0, np.nan)
            ).fillna(1.0)
    return pd.DataFrame(frames, index=volume.index)


def _volatility_features(log_ret: pd.Series, windows: list[int]) -> pd.DataFrame:
    """
    F. Realised volatility proxies (trailing rolling std of log returns).

    Annualised to per-bar scale for comparability.
    """
    frames = {}
    for w in windows:
        # Rolling std of log returns — pure trailing window
        frames[f"realized_vol_{w}"] = log_ret.rolling(w, min_periods=w).std()
    return pd.DataFrame(frames, index=log_ret.index)


def _time_features(index: pd.DatetimeIndex, cyclical: bool = True) -> pd.DataFrame:
    """
    G. Time-of-day and day-of-week features.

    Raw integers and optional sin/cos cyclical encoding.
    Cyclical encoding avoids the discontinuity at midnight/end-of-week.
    """
    feats = pd.DataFrame(index=index)
    feats["hour"] = index.hour
    feats["minute"] = index.minute
    feats["day_of_week"] = index.dayofweek  # 0=Mon … 4=Fri

    if cyclical:
        feats["hour_sin"] = np.sin(2 * np.pi * feats["hour"] / 24)
        feats["hour_cos"] = np.cos(2 * np.pi * feats["hour"] / 24)
        feats["minute_sin"] = np.sin(2 * np.pi * feats["minute"] / 60)
        feats["minute_cos"] = np.cos(2 * np.pi * feats["minute"] / 60)
        feats["dow_sin"] = np.sin(2 * np.pi * feats["day_of_week"] / 5)
        feats["dow_cos"] = np.cos(2 * np.pi * feats["day_of_week"] / 5)
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, feat_cfg: dict) -> pd.DataFrame:
    """
    Build a leakage-safe feature matrix plus the prediction target.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned OHLCV DataFrame indexed by timestamp.
    feat_cfg : dict
        The 'features' section of config.yaml.

    Returns
    -------
    pd.DataFrame
        Feature matrix with 'target' column appended.  All rows with NaN
        in any column (due to warmup) are dropped.

    Target definition
    -----------------
    target[t] = log(close[t+1] / close[t])
              = log_ret[t+1]
              = log_ret.shift(-1)[t]

    This is the quantity we want to predict.  The .shift(-1) is ONLY applied
    to the target column and nowhere else in this file.
    """
    lag_range = feat_cfg.get("lag_range", [1, 20])
    rolling_windows = feat_cfg.get("rolling_windows", [5, 10, 20, 60])
    momentum_windows = feat_cfg.get("momentum_windows", [3, 5, 10, 20])
    cyclical = feat_cfg.get("cyclical_time", True)

    max_lag = lag_range[1]

    # Base 1-min log return: ret[t] = log(close[t] / close[t-1])
    # NaN at session opens (gap > 5 min) to avoid contaminating features with
    # multi-hour overnight returns masquerading as 1-minute returns.
    log_ret = _log_returns(df["close"], df.index)

    parts = [
        _lagged_returns(log_ret, max_lag=max_lag),
        _rolling_return_stats(log_ret, rolling_windows),
        _momentum(log_ret, momentum_windows),
        _candle_structure(df),
        _volume_features(df["volume"], rolling_windows),
        _volatility_features(log_ret, rolling_windows),
        _time_features(df.index, cyclical=cyclical),
    ]
    feature_df = pd.concat(parts, axis=1)

    # ── TARGET CONSTRUCTION ───────────────────────────────────────────────────
    # target[t] = next bar's log return = log(close[t+1] / close[t])
    #
    # LEAKAGE NOTE: .shift(-1) is intentional HERE and ONLY here.
    # This moves the future return into the current row so the model can
    # learn to predict it.  The last row will be NaN (no future bar) and
    # is dropped below.  Feature columns must NEVER use .shift(-1).
    feature_df["target"] = log_ret.shift(-1)

    # Drop NaN rows from feature warmup AND the last row (no future target)
    n_before = len(feature_df)
    feature_df = feature_df.dropna()
    n_dropped = n_before - len(feature_df)
    logger.info(
        "Features built: %d rows (%d dropped for NaN warmup/tail).",
        len(feature_df), n_dropped,
    )

    return feature_df


def get_feature_names(model_df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (everything except 'target')."""
    return [c for c in model_df.columns if c != "target"]
