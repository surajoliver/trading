# VectorBT Backtesting Framework

Production-oriented framework for multi-universe, multi-strategy equity backtesting.

## Quick start

```bash
pip install -r requirements.txt
python main.py --config backtest.yaml --universe nifty50
```

Run several strategies:

```bash
python main.py --config backtest.yaml --universe nifty50 --strategies rsi momentum2
```

Optimize a parameter grid:

```bash
python main.py --config backtest.yaml --universe nifty50 --optimize
```

## Expected local data

Stock data:

`data/raw/Stock Data/{UNIVERSE}_data.pkl`

Universe constituents:

`data/raw/Indices/{UNIVERSE}.pkl`

The universe PKL must contain a `Yahoo` column. The loader accepts common PKL layouts for OHLCV data, including MultiIndex columns and per-symbol DataFrames.

## Weight convention

The strategy layer emits a target-weight matrix:

- `0.10` = target 10% position / buy
- `0.00` = exit
- `NaN` = no action / hold

Signals are sparse. There is no daily rebalance.

## Important implementation detail

Data is cleaned and resampled before the backtest window is truncated. A two-year warm-up is loaded before calculating indicators to avoid look-ahead/initialization artifacts.
