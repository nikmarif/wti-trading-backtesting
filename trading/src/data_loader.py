"""
data_loader.py — OHLCV ingestion, cleaning, and prototype data generation.

Architecture: a thin adapter pattern.
  - BaseLoader      abstract interface
  - CsvLoader       reads CSV files
  - ParquetLoader   reads Parquet files
  - HistDataLoader  reads HistData.com WTI 1-min CSVs (zipped or extracted),
                    with configurable year selection
  - SyntheticLoader generates realistic synthetic oil 1-min data for demos
  - load_and_clean()  top-level function called by the pipeline

To plug in a real data source (Polygon, broker API, etc.) implement BaseLoader
and pass it to load_and_clean(), or add a new Loader class below.
"""

from __future__ import annotations

import abc
import io
import logging
import zipfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .utils import get_logger, resolve_path

logger = get_logger(__name__)

# Expected canonical column names after loading
REQUIRED_COLS = {"timestamp", "open", "high", "low", "close", "volume"}


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class BaseLoader(abc.ABC):
    """All loaders must implement fetch() → raw DataFrame."""

    @abc.abstractmethod
    def fetch(self) -> pd.DataFrame:
        """Return a DataFrame with at minimum the REQUIRED_COLS columns."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Concrete loaders
# ─────────────────────────────────────────────────────────────────────────────

class CsvLoader(BaseLoader):
    """
    Load OHLCV data from a local CSV file.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.
    col_map : dict, optional
        Rename raw column names to the canonical names expected by the pipeline.
        E.g. {"Date": "timestamp", "Close": "close"}.
    """

    def __init__(self, path: str | Path, col_map: Optional[dict] = None):
        self.path = resolve_path(path)
        self.col_map = col_map or {}

    def fetch(self) -> pd.DataFrame:
        logger.info("Loading CSV: %s", self.path)
        df = pd.read_csv(self.path, low_memory=False)
        if self.col_map:
            df = df.rename(columns=self.col_map)
        return df


class ParquetLoader(BaseLoader):
    """Load OHLCV data from a local Parquet file."""

    def __init__(self, path: str | Path):
        self.path = resolve_path(path)

    def fetch(self) -> pd.DataFrame:
        logger.info("Loading Parquet: %s", self.path)
        return pd.read_parquet(self.path)


class HistDataLoader(BaseLoader):
    """
    Load HistData.com WTI 1-minute CSVs.

    HistData distributes one file per year, either:
      - Already extracted: data/wtiusd/DAT_ASCII_WTIUSD_M1_YYYY/DAT_ASCII_WTIUSD_M1_YYYY.csv
      - Zipped:            data/wtiusd/DAT_ASCII_WTIUSD_M1_YYYY.zip

    Format (semicolon-delimited, no header):
        YYYYMMDD HHMMSS;open;high;low;close;volume

    Parameters
    ----------
    folder : str or Path
        Directory containing the zip files and/or extracted sub-folders.
    years : list[int] or None
        Which years to load. If None, loads all years found in folder.
    """

    # HistData column layout (no header in file)
    _COLS = ["timestamp", "open", "high", "low", "close", "volume"]

    def __init__(self, folder: str | Path, years: Optional[List[int]] = None):
        self.folder = resolve_path(folder)
        self.years = years  # None = load all

    def _year_files(self) -> list[tuple[int, Path]]:
        """
        Return list of (year, path) for all available years, sorted ascending.
        Prefers already-extracted CSV; falls back to zip.
        """
        available: dict[int, Path] = {}

        # Scan for extracted folders
        for subdir in sorted(self.folder.iterdir()):
            if subdir.is_dir() and "WTIUSD_M1_" in subdir.name:
                try:
                    year = int(subdir.name[-4:])
                except ValueError:
                    continue
                csv = subdir / f"{subdir.name}.csv"
                if csv.exists():
                    available[year] = csv

        # Scan for zip files (only for years not already extracted)
        for zp in sorted(self.folder.glob("DAT_ASCII_WTIUSD_M1_????.zip")):
            try:
                year = int(zp.stem[-4:])
            except ValueError:
                continue
            if year not in available:
                available[year] = zp

        if self.years is not None:
            missing = set(self.years) - set(available)
            if missing:
                raise FileNotFoundError(
                    f"Requested years not found in {self.folder}: {sorted(missing)}"
                )
            available = {y: p for y, p in available.items() if y in self.years}

        return sorted(available.items())

    def _read_csv_bytes(self, data: bytes, year: int) -> pd.DataFrame:
        """Parse raw CSV bytes into a DataFrame."""
        df = pd.read_csv(
            io.BytesIO(data),
            sep=";",
            header=None,
            names=self._COLS,
            dtype={"open": float, "high": float, "low": float,
                   "close": float, "volume": float},
        )
        return df

    def _load_one(self, year: int, path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".zip":
            logger.info("Unzipping year %d: %s", year, path.name)
            with zipfile.ZipFile(path) as zf:
                # Find the CSV inside the zip
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    raise ValueError(f"No CSV found inside {path}")
                data = zf.read(csv_names[0])
            return self._read_csv_bytes(data, year)
        else:
            logger.info("Reading year %d: %s", year, path.name)
            df = pd.read_csv(
                path, sep=";", header=None, names=self._COLS,
                dtype={"open": float, "high": float, "low": float,
                       "close": float, "volume": float},
            )
            return df

    def fetch(self) -> pd.DataFrame:
        year_files = self._year_files()
        if not year_files:
            raise FileNotFoundError(
                f"No HistData WTI files found in {self.folder}. "
                "Expected files like DAT_ASCII_WTIUSD_M1_YYYY.zip or extracted folders."
            )

        years_loaded = [y for y, _ in year_files]
        logger.info("Loading years: %s", years_loaded)

        chunks = [self._load_one(y, p) for y, p in year_files]
        df = pd.concat(chunks, ignore_index=True)
        logger.info("Raw rows loaded: %d", len(df))
        return df


class SyntheticLoader(BaseLoader):
    """
    Generate synthetic 1-minute WTI-like OHLCV data using a GBM price process.

    This is a *prototype only* — use real data for any actual research.
    A weak, decaying predictive signal is injected so the pipeline has
    something to find without being trivially predictable.
    """

    def __init__(self, n_bars: int = 50_000, seed: int = 42):
        self.n_bars = n_bars
        self.seed = seed

    def fetch(self) -> pd.DataFrame:
        logger.info("Generating %d synthetic 1-min bars (seed=%d)", self.n_bars, self.seed)
        rng = np.random.default_rng(self.seed)
        n = self.n_bars

        # GBM log returns: WTI rough params (annualised vol ~35%, 252*390 bars/yr)
        sigma_per_bar = 0.35 / np.sqrt(252 * 390)
        mu_per_bar = 0.0  # zero drift

        # Inject a weak autocorrelated signal (momentum-ish, decays fast)
        # Signal at bar t slightly predicts return at t+1 — realistic but tiny.
        signal = rng.standard_normal(n) * sigma_per_bar * 0.3
        signal = pd.Series(signal).ewm(span=5).mean().values  # smooth the signal
        noise = rng.standard_normal(n) * sigma_per_bar

        log_rets = mu_per_bar + signal + noise  # total log return per bar

        # Reconstruct price level starting near WTI spot ~80 USD
        log_price = np.log(80.0) + np.cumsum(log_rets)
        close = np.exp(log_price)

        # OHLC construction: intra-bar hi/lo drawn from bar's vol
        intra_vol = sigma_per_bar * np.abs(rng.standard_normal(n)) * 0.5
        high = close * np.exp(intra_vol)
        low = close * np.exp(-intra_vol)
        # Open: previous bar's close (continuous contract assumption)
        open_ = np.roll(close, 1)
        open_[0] = close[0]

        # Clip high/low so they always bracket open and close
        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))

        # Volume: log-normal with time-of-day seasonality
        base_vol = rng.lognormal(mean=10.0, sigma=0.5, size=n).astype(int) + 1
        # Fake intraday shape: high near open/close of trading session
        minute_idx = np.arange(n) % 390  # 390 min/trading day
        tod_factor = 1 + 0.5 * np.exp(-((minute_idx - 0) ** 2) / (2 * 30**2))
        tod_factor += 0.5 * np.exp(-((minute_idx - 389) ** 2) / (2 * 30**2))
        volume = (base_vol * tod_factor).astype(int)

        # Timestamps: start 2022-01-03 09:30 ET, skip weekends naively
        timestamps = pd.date_range(
            start="2022-01-03 09:30:00", periods=n, freq="1min", tz="UTC"
        )

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })
        return df


# ─────────────────────────────────────────────────────────────────────────────
# Placeholder adapters — fill these in to connect a real data source
# ─────────────────────────────────────────────────────────────────────────────

class PolygonLoader(BaseLoader):
    """
    Placeholder: fetch 1-min WTI data from Polygon.io.

    To implement:
      pip install polygon-api-client
      from polygon import RESTClient
      client = RESTClient(api_key=...)
      aggs = client.list_aggs(ticker="CL", multiplier=1, timespan="minute", ...)
      Convert to DataFrame with REQUIRED_COLS and return.
    """

    def __init__(self, ticker: str, api_key: str, from_date: str, to_date: str):
        self.ticker = ticker
        self.api_key = api_key
        self.from_date = from_date
        self.to_date = to_date

    def fetch(self) -> pd.DataFrame:
        raise NotImplementedError(
            "PolygonLoader is a placeholder. "
            "See the docstring for implementation instructions."
        )


class BrokerApiLoader(BaseLoader):
    """
    Placeholder: fetch 1-min bars from a broker REST API (e.g. Interactive Brokers,
    Alpaca, TradeStation).

    To implement: replace fetch() with the broker SDK call, ensure the returned
    DataFrame has the canonical REQUIRED_COLS columns.
    """

    def fetch(self) -> pd.DataFrame:
        raise NotImplementedError("BrokerApiLoader is a placeholder.")


# ─────────────────────────────────────────────────────────────────────────────
# Data cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_ohlcv(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    timezone: Optional[str] = "UTC",
    reindex_to_1min: bool = True,
) -> pd.DataFrame:
    """
    Validate, clean, and standardise a raw OHLCV DataFrame.

    Steps
    -----
    1. Rename timestamp column → 'timestamp'; parse to datetime.
    2. Set timestamp as index; sort ascending.
    3. Handle timezone.
    4. Drop duplicates.
    5. Validate OHLC ordering (high >= max(open,close), low <= min(open,close)).
    6. Report missing 1-minute bars.
    7. Optionally reindex to a full 1-minute grid.
    8. Enforce positive prices and non-negative volume.

    Returns a clean DataFrame indexed by timestamp with the canonical REQUIRED_COLS
    minus 'timestamp' (which is the index).
    """
    df = df.copy()

    # ── 1. Timestamp ──────────────────────────────────────────────────────────
    if timestamp_col not in df.columns and df.index.name == timestamp_col:
        df = df.reset_index()
    if timestamp_col not in df.columns:
        raise ValueError(
            f"Timestamp column '{timestamp_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Try HistData format first (YYYYMMDD HHMMSS = 15 chars, no separators in date part)
    sample = df[timestamp_col].dropna().iloc[0] if len(df) else ""
    if isinstance(sample, str) and len(sample) == 15 and " " in sample:
        df["timestamp"] = pd.to_datetime(
            df[timestamp_col], format="%Y%m%d %H%M%S", utc=(timezone == "UTC")
        )
    else:
        df["timestamp"] = pd.to_datetime(df[timestamp_col], utc=(timezone == "UTC"))
    if timezone and timezone != "UTC":
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(timezone)
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert(timezone)

    # Ensure required OHLCV columns exist
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in ohlcv_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[["timestamp"] + ohlcv_cols].copy()
    df = df.set_index("timestamp").sort_index()

    # ── 2. Duplicates ─────────────────────────────────────────────────────────
    n_before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning("Dropped %d duplicate timestamps.", n_dropped)

    # ── 3. Numeric coercion ───────────────────────────────────────────────────
    for col in ohlcv_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 4. Sanity: prices must be positive ───────────────────────────────────
    bad_price = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if bad_price.any():
        logger.warning("Dropping %d rows with non-positive prices.", bad_price.sum())
        df = df[~bad_price]

    df["volume"] = df["volume"].clip(lower=0)

    # ── 5. OHLC ordering check (log only, don't drop — data sources vary) ────
    ohlc_invalid = (
        (df["high"] < df[["open", "close"]].max(axis=1)) |
        (df["low"] > df[["open", "close"]].min(axis=1))
    )
    if ohlc_invalid.any():
        logger.warning(
            "%d rows have high < max(open,close) or low > min(open,close). "
            "Clamping to valid range.",
            ohlc_invalid.sum(),
        )
        df["high"] = df[["high", "open", "close"]].max(axis=1)
        df["low"] = df[["low", "open", "close"]].min(axis=1)

    # ── 6. Missing bar report ─────────────────────────────────────────────────
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="1min",
                             tz=df.index.tz)
    missing_bars = full_idx.difference(df.index)
    if len(missing_bars):
        pct = 100 * len(missing_bars) / len(full_idx)
        logger.info(
            "Missing 1-min bars: %d / %d (%.2f%%). First few: %s",
            len(missing_bars), len(full_idx), pct,
            list(missing_bars[:5]),
        )

    # ── 7. Reindex ────────────────────────────────────────────────────────────
    if reindex_to_1min and len(missing_bars):
        df = df.reindex(full_idx)
        # Forward-fill OHLC (price unchanged during gap), zero-fill volume
        df[["open", "high", "low", "close"]] = (
            df[["open", "high", "low", "close"]].ffill()
        )
        df["volume"] = df["volume"].fillna(0)
        logger.info("Reindexed to full 1-min grid (%d bars).", len(df))

    # Final NaN drop (e.g. very start of series before any close)
    df = df.dropna(subset=["close"])

    logger.info("Clean data: %d bars | %s → %s",
                len(df), df.index.min(), df.index.max())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ─────────────────────────────────────────────────────────────────────────────

def load_and_clean(cfg: dict) -> pd.DataFrame:
    """
    Load raw OHLCV data according to config, clean it, and save processed Parquet.

    Parameters
    ----------
    cfg : dict
        The 'data' section of config.yaml.

    Returns
    -------
    pd.DataFrame
        Clean OHLCV DataFrame indexed by timestamp.
    """
    data_cfg = cfg["data"]

    if data_cfg.get("use_synthetic", False):
        loader: BaseLoader = SyntheticLoader(
            n_bars=data_cfg.get("synthetic_n_bars", 50_000),
            seed=cfg.get("model", {}).get("random_state", 42),
        )
    elif data_cfg.get("source") == "histdata":
        loader = HistDataLoader(
            folder=resolve_path(data_cfg["histdata_folder"]),
            years=data_cfg.get("years", None),
        )
    else:
        raw_path = resolve_path(data_cfg["raw_path"])
        if raw_path.suffix.lower() == ".parquet":
            loader = ParquetLoader(raw_path)
        else:
            loader = CsvLoader(raw_path)

    raw = loader.fetch()

    clean = clean_ohlcv(
        raw,
        timestamp_col=data_cfg.get("timestamp_col", "timestamp"),
        timezone=data_cfg.get("timezone", "UTC"),
        reindex_to_1min=data_cfg.get("reindex_to_1min", True),
    )

    # Persist processed data
    processed_path = resolve_path(data_cfg["processed_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(processed_path)
    logger.info("Saved processed data → %s", processed_path)

    return clean
