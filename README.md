# WTI Trading Backtesting

A modular backtesting framework for WTI crude oil 1-minute bars. Plug in any strategy — technical, ML-based, or your own — and get consistent performance metrics and trade charts out the other side.

---

## What this is

A clean engine for answering the question: *does this strategy make money on WTI 1-minute data after costs?*

Strategies are interchangeable. The same backtest engine, cost model, and reporting run regardless of whether the signal comes from a moving-average crossover, an RSI threshold, or an XGBoost model trained on 50 features. You write the signal logic; the framework handles the rest.

---

## Strategies

| Name | Type | Description |
|---|---|---|
| `ma_crossover` | Technical | Fast/slow SMA or EMA crossover |
| `ema_crossover` | Technical | Same as above with EMA |
| `rsi` | Technical | RSI mean-reversion (oversold/overbought) |
| `bollinger` | Technical | Bollinger Band mean-reversion |
| XGBoost pipeline | ML | Walk-forward XGBoost regression on OHLCV features |

Adding your own takes about 10 lines of code — see [Adding a strategy](#adding-a-strategy).

---

## Backtest engine

- **No lookahead**: position at bar `t` is computed strictly from data available at or before bar `t`
- **Transaction costs**: charged on every position change — `cost_per_trade + slippage`
- **Metrics**: total return, annualised Sharpe, max drawdown, Calmar, hit rate, turnover, position breakdown
- **Trade chart**: price with shaded holding regions, entry/exit markers, per-trade P&L labels, equity curve

---

## Quick start

### Install

```bash
pip install -r requirements.txt
```

### Point at your data

Edit `config/config.yaml`:

```yaml
data:
  use_synthetic: false
  raw_path: "data/raw/wti_1m.csv"
```

Or use the built-in synthetic data for a demo:

```yaml
data:
  use_synthetic: true
  synthetic_n_bars: 50000
```

### Run a strategy

```bash
# List available strategies
python scripts/run_strategy_backtest.py --list

# Run with defaults
python scripts/run_strategy_backtest.py --strategy ma_crossover
python scripts/run_strategy_backtest.py --strategy rsi
python scripts/run_strategy_backtest.py --strategy bollinger

# Override parameters
python scripts/run_strategy_backtest.py --strategy ma_crossover --params fast=3 slow=30 ma_type=ema
python scripts/run_strategy_backtest.py --strategy rsi --params period=7 oversold=25 overbought=75

# Save a trade chart
python scripts/run_strategy_backtest.py --strategy ma_crossover --plot
python scripts/run_strategy_backtest.py --strategy ma_crossover --plot --plot-bars 200
```

### Run the XGBoost pipeline

```bash
python scripts/run_pipeline.py
# rolling-window CV instead of expanding:
python scripts/run_pipeline.py --cv rolling
```

---

## Adding a strategy

1. Open `src/strategies/technical.py`
2. Subclass `BaseStrategy` and implement `generate_positions()`
3. Register it in `REGISTRY`

```python
class MyStrategy(BaseStrategy):
    def __init__(self, period: int = 10):
        self.period = period

    def generate_positions(self, ohlcv: pd.DataFrame) -> pd.Series:
        # return a Series of +1 (long), -1 (short), 0 (flat)
        # no lookahead — only use data up to bar t
        ...

REGISTRY["my_strategy"] = (MyStrategy, {"period": 10})
```

Then run it:
```bash
python scripts/run_strategy_backtest.py --strategy my_strategy --plot
```

---

## Data format

CSV or Parquet with these columns (names configurable in `config/config.yaml`):

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | Bar end time (1-min bars) |
| `open` | float | Bar open |
| `high` | float | Bar high |
| `low` | float | Bar low |
| `close` | float | Bar close |
| `volume` | float | Bar volume |

To plug in a live data source, subclass `BaseLoader` in `src/data_loader.py` and implement `fetch()`.

---

## Outputs

All outputs land in `data/artifacts/`:

| File | Description |
|---|---|
| `trades_<strategy>.png` | Trade chart: price, entries, exits, per-trade P&L, equity curve |
| `cumulative_returns.png` | Strategy vs buy-and-hold (XGBoost pipeline) |
| `oos_predictions.parquet` | OOS predictions with timestamps (XGBoost pipeline) |
| `metrics.json` | Regression metrics per fold and overall (XGBoost pipeline) |
| `threshold_sweep.csv` | Sharpe and return vs signal threshold (XGBoost pipeline) |
| `model_last_fold.pkl` | Trained XGBoost model (last fold) |
| `feature_importance.png` | Top-30 XGBoost feature importances |

---

## Limitations

- **1-minute returns are very noisy.** Signal-to-noise ratio is extremely low at this frequency.
- **Transaction costs are model inputs, not measurements.** Real costs depend on execution venue, size, and market conditions.
- **No market impact modelling.** Assumes you are small relative to the market.
- **Synthetic data is not real data.** The demo generator produces plausible-looking prices but not realistic dynamics.
- **This is research code, not production trading advice.**

---

## Project structure

```
trading/
├── config/
│   └── config.yaml               # all parameters
├── data/
│   ├── raw/                      # place your CSV/Parquet here
│   ├── processed/                # cleaned data (auto-generated)
│   └── artifacts/                # outputs (auto-generated)
├── src/
│   ├── strategies/
│   │   ├── base.py               # BaseStrategy abstract class
│   │   └── technical.py          # MACrossover, RSI, BollingerBand + REGISTRY
│   ├── backtest.py               # backtest engine (threshold + position-based)
│   ├── plotting.py               # trade chart + all other figures
│   ├── data_loader.py            # OHLCV loading, cleaning, synthetic generator
│   ├── feature_engineering.py    # leakage-safe features (XGBoost pipeline)
│   ├── validation.py             # expanding/rolling walk-forward CV
│   ├── train.py                  # XGBoost walk-forward training loop
│   ├── metrics.py                # RMSE, MAE, correlation, sign accuracy
│   └── utils.py                  # config loading, logging, path helpers
├── scripts/
│   ├── run_strategy_backtest.py  # strategy runner (technical + custom)
│   └── run_pipeline.py           # XGBoost end-to-end pipeline
├── tests/
│   └── test_strategies.py        # unit tests (19 tests)
├── requirements.txt
└── README.md
```
