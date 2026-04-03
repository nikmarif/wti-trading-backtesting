# WTI Trading Backtesting

XGBoost regression pipeline for predicting next-1-minute crude oil log returns, plus a modular strategy backtester for testing rule-based strategies (MA crossover, RSI, Bollinger Bands, and any custom strategy you add).

---

## Project objective

Predict `target[t] = log(close[t+1] / close[t])` using only information available at the end of bar `t`.  The goal is to evaluate whether any predictive signal exists in 1-minute OHLCV data, and whether that signal survives transaction costs in a simple backtest.

A secondary goal is to provide a clean, modular framework for comparing the ML signal against simple technical strategies on the same data and cost assumptions.

---

## Expected data format

A CSV or Parquet file with these columns (names configurable in `config/config.yaml`):

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime string or epoch | Bar end time (1-min bars) |
| `open` | float | Bar open price |
| `high` | float | Bar high price |
| `low` | float | Bar low price |
| `close` | float | Bar close price |
| `volume` | float | Bar volume (contracts or USD) |

The pipeline sorts ascending, deduplicates, and optionally reindexes to a full 1-minute grid.

---

## How the target is defined

```
log_ret[t]  = log(close[t] / close[t-1])   # 1-min return ending at bar t
target[t]   = log_ret.shift(-1)[t]          # = log(close[t+1] / close[t])
```

The model learns to predict `target[t]` from features computed at or before bar `t`.

---

## Leakage precautions

| Risk | How it's handled |
|---|---|
| Rolling windows | Only trailing `.rolling(w)` — never `center=True` |
| Lagged features | All lags ≥ 1 bar |
| Target construction | `.shift(-1)` applied **only** to the target column, clearly marked in `feature_engineering.py` |
| Walk-forward CV | No future data seen during training — expanding window, no shuffle |
| Backtest | Operates on OOS predictions only; position at `t` uses prediction at `t`, realises return at `t+1` |
| Scaling | Not used for tree models; if added, must be fit inside each fold on train only |

---

## Walk-forward validation

The default strategy is an **expanding window**:

```
Fold 0:  train [0, min_train_bars)        val [min_train_bars, min_train_bars + val_bars)
Fold 1:  train [0, min_train_bars + step)  val [min_train_bars + step, ...)
...
```

A rolling-window variant is also available (`--cv rolling`).

Parameters are set in `config.yaml`:
- `min_train_bars`: minimum training bars (default 15 000, ~10 days)
- `val_bars`: validation window size (default 5 000)
- `step_bars`: advance per fold (default 2 500)

---

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Point at your data

Edit `config/config.yaml`:

```yaml
data:
  use_synthetic: false          # set to false for real data
  raw_path: "data/raw/wti_1m.csv"
```

Or to use the built-in synthetic data generator for a demo run:

```yaml
data:
  use_synthetic: true
  synthetic_n_bars: 50000
```

### 3. Run the ML pipeline

```bash
python scripts/run_pipeline.py
# or with rolling-window CV:
python scripts/run_pipeline.py --cv rolling
# or with a custom config:
python scripts/run_pipeline.py --config my_config.yaml
```

### 4. Run a rule-based strategy backtest

```bash
# List available strategies
python scripts/run_strategy_backtest.py --list

# Run with defaults
python scripts/run_strategy_backtest.py --strategy ma_crossover
python scripts/run_strategy_backtest.py --strategy rsi
python scripts/run_strategy_backtest.py --strategy bollinger

# Override strategy parameters
python scripts/run_strategy_backtest.py --strategy ma_crossover --params fast=3 slow=30 ma_type=ema
python scripts/run_strategy_backtest.py --strategy rsi --params period=7 oversold=25 overbought=75

# Save a trade chart (shows entries, exits, and P&L per trade)
python scripts/run_strategy_backtest.py --strategy ma_crossover --plot
python scripts/run_strategy_backtest.py --strategy ma_crossover --plot --plot-bars 200
```

---

## How to add a new strategy

1. Open `src/strategies/technical.py`.
2. Subclass `BaseStrategy` and implement `generate_positions(ohlcv) -> pd.Series` returning `+1`, `-1`, or `0` for each bar.
3. Add an entry to the `REGISTRY` dict at the bottom of the file.

```python
class MyStrategy(BaseStrategy):
    def __init__(self, period: int = 10):
        self.period = period

    def generate_positions(self, ohlcv: pd.DataFrame) -> pd.Series:
        # your logic here — no lookahead allowed
        ...

REGISTRY["my_strategy"] = (MyStrategy, {"period": 10})
```

Then run it immediately:
```bash
python scripts/run_strategy_backtest.py --strategy my_strategy --plot
```

---

## How to plug in a real oil data source

See `src/data_loader.py`.  Three patterns:

**Option A — CSV/Parquet (simplest):**
Drop your file into `data/raw/` and set `raw_path` in config.

**Option B — Polygon.io:**
Fill in `PolygonLoader.fetch()` in `data_loader.py`.  The skeleton is already there.

**Option C — Any other source:**
Subclass `BaseLoader`, implement `fetch() → pd.DataFrame` returning a DataFrame with the canonical columns, and pass it to `clean_ohlcv()`.

---

## Outputs

All outputs land in `data/artifacts/`:

| File | Description |
|---|---|
| `oos_predictions.parquet` | All out-of-sample predictions with timestamps |
| `metrics.json` | Regression metrics per fold and overall |
| `model_fold_N.pkl` | Trained XGBoost model for fold N |
| `model_last_fold.pkl` | Last-fold model (used for feature importance) |
| `feature_names.json` | Ordered list of feature names |
| `actual_vs_predicted.png` | Scatter + time-series of predictions vs actuals |
| `cumulative_returns.png` | Strategy cumulative return vs buy-and-hold |
| `prediction_histogram.png` | Distribution of predictions with threshold lines |
| `feature_importance.png` | Top-30 XGBoost feature importances |
| `fold_metrics.png` | Per-fold correlation and sign accuracy |
| `threshold_sweep.png` | Sharpe and return vs threshold |
| `threshold_sweep.csv` | Numerical threshold sweep table |
| `trades_<strategy>.png` | Trade chart: price, entry/exit markers, per-trade P&L, equity curve |
| `oil_1m_clean.parquet` | Cleaned OHLCV data (in `data/processed/`) |

---

## How to interpret outputs

- **Correlation**: A value of 0.01–0.05 is typical for 1-minute financial returns; 0.10+ is exceptional.
- **Sign accuracy**: 50% = random; 52–53% can be tradeable if costs are low.
- **Sharpe**: > 0.5 after costs is worth investigating further.
- **Max drawdown**: In log-return units; multiply by notional to get dollar drawdown.
- **Threshold sweep**: Lower thresholds trade more but pay more costs; higher thresholds trade less but may miss edge.

---

## Limitations

- **1-minute returns are very noisy.** The signal-to-noise ratio is extremely low.
- **Predictive power may be tiny.** Even a correlation of 0.02 can be statistically significant over 50k bars but practically irrelevant.
- **Transaction costs are model inputs, not measurements.** Real costs depend on your execution venue, size, and market conditions.
- **No market impact modelling.** Assume we are small relative to the market.
- **Synthetic data is not real data.** The prototype loader generates plausible-looking prices but the dynamics are not realistic.
- **This is research code, not production trading advice.**

---

## Project structure

```
trading/
├── config/
│   └── config.yaml          # all parameters
├── data/
│   ├── raw/                 # place your raw CSV/Parquet here
│   ├── processed/           # cleaned data (auto-generated)
│   └── artifacts/           # models, predictions, plots (auto-generated)
├── src/
│   ├── data_loader.py       # OHLCV loading, cleaning, synthetic generator
│   ├── feature_engineering.py  # leakage-safe feature construction
│   ├── validation.py        # expanding/rolling walk-forward CV
│   ├── train.py             # XGBoost walk-forward training loop
│   ├── backtest.py          # threshold backtest + position-based backtest + sweep
│   ├── metrics.py           # RMSE, MAE, correlation, sign accuracy
│   ├── plotting.py          # all figures, including trade chart
│   ├── utils.py             # config loading, logging, path helpers
│   └── strategies/
│       ├── base.py          # BaseStrategy abstract class
│       └── technical.py     # MACrossover, RSI, BollingerBand + REGISTRY
├── scripts/
│   ├── run_pipeline.py      # end-to-end ML pipeline runner
│   └── run_strategy_backtest.py  # standalone rule-based strategy runner
├── tests/
│   └── test_strategies.py  # unit tests for strategies + backtest engine
├── notebooks/
│   └── exploration.ipynb    # ad-hoc analysis
├── requirements.txt
└── README.md
```
