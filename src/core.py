"""Core backtesting engine, data loading, and metrics."""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import yfinance as yf
import vectorbt as vbt
from src.utils import safe_float, get_trading_dates, progress_bar

# ============================================================
# DATA LOADER
# ============================================================

class DataLoader:
    """Load and prepare market data for backtesting."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.root = Path(__file__).resolve().parent.parent
        self.data_dir = self.root / config.get("stock_data_dir", "data/raw/Stock Data")
        self.index_dir = self.root / config.get("index_dir", "data/raw/Indices")
        self.max_ffill_days = config.get("max_ffill_days", 2)
    
    def resolve_universe(self, universe: Union[str, List[str]]) -> tuple[Optional[str], List[str]]:
        """Resolve universe name to list of tickers."""
        if isinstance(universe, str):
            path = self.index_dir / f"{universe}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Universe {universe} file not found: {path}")
            df = pd.read_csv(path)
            if "Yahoo" not in df.columns:
                raise ValueError(f"'Yahoo' column not found in {path}")
            tickers = df["Yahoo"].dropna().astype(str).str.strip().unique().tolist()
            return universe, tickers
        elif isinstance(universe, (list, tuple, np.ndarray)):
            tickers = list(universe)
            if not tickers:
                raise ValueError("Universe is empty.")
            return None, tickers
        else:
            raise TypeError("Universe must be an index name or ticker list.")
    
    def load(self, universe: Union[str, List[str]], start: str, end: Optional[str] = None) -> pd.DataFrame:
        """Load price data from source."""
        universe_name, tickers = self.resolve_universe(universe)

        end_date = ( pd.Timestamp.today().normalize() if end is None else pd.Timestamp(end) )
        # Add warm-up period
        data_start = pd.Timestamp(start) - pd.DateOffset(years=self.config.get("warmup_years", 2))
        
        # Load data
        if self.config.get("source") == "pkl":
            if universe_name is None:
                raise ValueError("PKL source requires a named universe.")
            path = self.data_dir / f"{universe_name}_data.pkl"
            if not path.exists():
                raise FileNotFoundError(f"PKL file not found: {path}")
            obj = pd.read_pickle(path)
            prices = obj["data"] if isinstance(obj, dict) else obj
            if isinstance(prices.columns, pd.MultiIndex):
                prices = prices["Close"]
            prices = prices[prices.columns.intersection(tickers)]
            prices = prices.loc[data_start:end_date]
            
        elif self.config.get("source") == "yahoo":
            prices = yf.download(
                tickers,
                start=data_start,
                end=end_date,
                auto_adjust=True,
                progress=False,
            )

            if prices.empty:
                raise ValueError("Yahoo Finance returned no data.")

            if isinstance(prices.columns, pd.MultiIndex):
                prices = prices["Close"]
            else:
                prices = prices[["Close"]]

            prices = prices.reindex(columns=tickers)
        
        # Clean
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()
        prices = prices.loc[~prices.index.duplicated()]
        prices = prices.replace([np.inf, -np.inf, 0], np.nan)
        prices = prices.mask(prices <= 0, np.nan)
        
        # Forward fill with limit
        prices = prices.ffill(limit=self.max_ffill_days)
        prices = prices.dropna(axis=1, how='all')
        
        if prices.empty:
            raise ValueError("No valid price data.")
        
        return prices

# ============================================================
# BACKTEST ENGINE
# ============================================================

@dataclass
class BacktestResult:
    strategy: str
    portfolio: vbt.Portfolio
    target_weights: pd.DataFrame
    diagnostics: pd.DataFrame
    trade_log: pd.DataFrame
    value: pd.Series | None = None
    metrics: dict | None = None

def run_backtest(
    close: pd.DataFrame,
    target_weights: pd.DataFrame,
    config: Dict[str, Any],
    strategy_name: str = "strategy"
) -> BacktestResult:
    """Run vectorbt backtest with given weights."""
    # Align data
    close, target_weights = close.align(target_weights, join="left", axis=0)
    target_weights = target_weights.reindex(columns=close.columns, fill_value=0)
    
    
    # Run portfolio
    pf = vbt.Portfolio.from_orders(
        close=close,
        size=target_weights,
        size_type="targetpercent",
        init_cash=config.get("initial_cash", 1_000_000),
        fees=config.get("fees", 0.0025),
        slippage=config.get("slippage", 0.0),
        freq="1D",
        cash_sharing=config.get("cash_sharing", True),
        group_by=True,
    )
    value = pf.value()
    metrics = calculate_metrics(pf, value=value)
    
    # Build diagnostics
    diagnostics = build_diagnostics(close, target_weights, value)
    trade_log = get_trade_log(pf)
    
    return BacktestResult(
        strategy=strategy_name,
        portfolio=pf,
        target_weights=target_weights,
        diagnostics=diagnostics,
        trade_log=trade_log,
        value=value,
        metrics=metrics,
    )

def build_diagnostics(close, weights, value):
    
    rows = weights.stack().rename("TargetWeight").reset_index()
    rows.columns = ["Date", "Symbol", "TargetWeight"]

    rows["Action"] = np.where(
        rows["TargetWeight"].eq(0),
        "SELL",
        "BUY"
    )

    rows["Price"] = [
        close.at[dt, sym]
        for dt, sym in zip(rows["Date"], rows["Symbol"])
    ]

    # value = pf.value()
    rows["PortfolioValue"] = rows["Date"].map(value)

    return rows[
        ["Date", "Symbol", "Action", "TargetWeight", "Price", "PortfolioValue"]
    ]


def get_trade_log(pf: vbt.Portfolio) -> pd.DataFrame:
    """Extract detailed trade log from portfolio."""
    try:
        trades = pf.trades.records_readable.copy()
        if trades.empty:
            return trades
        rename = {
            "Entry Timestamp": "EntryDate",
            "Exit Timestamp": "ExitDate",
            "Entry Price": "EntryPrice",
            "Exit Price": "ExitPrice",
            "PnL": "PnL",
            "Return": "Return",
            "Direction": "Direction",
            "Size": "Size",
            "Status": "Status"
        }
        return trades.rename(columns=rename)
    except Exception:
        return pd.DataFrame()

# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    pf: vbt.Portfolio,
    benchmark: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    value: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """Calculate comprehensive performance metrics."""
    value = pf.value() if value is None else value
    if len(value) < 2:
        return {}
    
    ret = value.pct_change().dropna()
    if ret.empty:
        return {}
    
    # Returns
    total_return = value.iloc[-1] / value.iloc[0] - 1
    years = (value.index[-1] - value.index[0]).days / 365.25
    annualized = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan
    
    # Risk-adjusted returns
    ret_mean = ret.mean()
    ret_std = ret.std()
    sharpe = (ret_mean - risk_free_rate) / ret_std * np.sqrt(252) if ret_std > 0 else np.nan
    
    downside = ret[ret < 0].std()
    sortino = (ret_mean - risk_free_rate) / downside * np.sqrt(252) if downside and downside > 0 else np.nan
    
    # Drawdown
    peak = value.cummax()
    dd = (value - peak) / peak
    max_dd = abs(dd.min()) if not dd.empty else np.nan
    max_dd_duration = 0
    if not dd.empty and dd.min() < 0:
        trough_date = dd.idxmin()
        running_max = value.cummax().loc[:trough_date]
        peak_date = running_max.idxmax()
        max_dd_duration = (trough_date - peak_date).days
    
    calmar = annualized / max_dd if max_dd and max_dd > 0 else np.nan
    ulcer = np.sqrt((dd ** 2).mean()) if not dd.empty else np.nan
    
    # Gain-to-Pain ratio
    gains = ret[ret > 0].sum()
    losses = abs(ret[ret < 0].sum())
    gain_pain = gains / losses if losses != 0 else np.nan
    
    # Trade statistics
    trades = pf.trades.records_readable
    trade_metrics = {}
    if len(trades) > 0:
        pnl = pd.to_numeric(trades.get("PnL", pd.Series(dtype=float)), errors="coerce").dropna()
        if len(pnl) > 0:
            wins = pnl[pnl > 0]
            losses_trades = pnl[pnl < 0]
            win_rate = len(wins) / len(pnl)
            profit_factor = wins.sum() / abs(losses_trades.sum()) if len(losses_trades) > 0 and losses_trades.sum() != 0 else np.inf
            avg_win = wins.mean() if len(wins) > 0 else np.nan
            avg_loss = losses_trades.mean() if len(losses_trades) > 0 else np.nan
            expectancy = pnl.mean()
            total_trades = len(pnl)
            trade_freq = total_trades / years if years > 0 else np.nan
            
            # Risk of ruin (empirical proxy)
            p = win_rate
            b = abs(avg_win / avg_loss) if pd.notna(avg_loss) and avg_loss != 0 else np.nan
            risk_ruin = ((1-p)/(1+p*b))**100 if pd.notna(p) and pd.notna(b) and b > 0 else np.nan
            
            trade_metrics = {
                "Win Rate": win_rate,
                "Profit Factor": profit_factor,
                "Average Win": avg_win,
                "Average Loss": avg_loss,
                "Expectancy": expectancy,
                "Total Trades": total_trades,
                "Trade Frequency / Year": trade_freq,
                "Risk of Ruin": risk_ruin,
            }
        else:
            trade_metrics = {
                "Win Rate": np.nan,
                "Profit Factor": np.nan,
                "Average Win": np.nan,
                "Average Loss": np.nan,
                "Expectancy": np.nan,
                "Total Trades": 0,
                "Trade Frequency / Year": np.nan,
                "Risk of Ruin": np.nan,
            }
    else:
        trade_metrics = {
            "Win Rate": np.nan,
            "Profit Factor": np.nan,
            "Average Win": np.nan,
            "Average Loss": np.nan,
            "Expectancy": np.nan,
            "Total Trades": 0,
            "Trade Frequency / Year": np.nan,
            "Risk of Ruin": np.nan,
        }
    
    metrics = {
        "Total Return": total_return,
        "Annualized Return": annualized,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Max Drawdown": max_dd,
        "Max Drawdown Duration (days)": max_dd_duration,
        "Ulcer Index": ulcer,
        "Gain-to-Pain Ratio": gain_pain,
        **trade_metrics
    }
    
    # Benchmark comparison
    if benchmark is not None:
        b = benchmark.reindex(value.index).ffill().dropna()
        if len(b) > 1:
            bench_return = b.iloc[-1] / b.iloc[0] - 1
            metrics["Benchmark Return"] = bench_return
            metrics["Excess Return"] = total_return - bench_return
    
    return metrics

def monthly_returns(pf: vbt.Portfolio, value: Optional[pd.Series] = None) -> pd.DataFrame:
    """Calculate monthly return heatmap."""
    value = pf.value() if value is None else value
    ret = value.resample("ME").last().pct_change().dropna()
    if ret.empty:
        return pd.DataFrame()
    df = ret.to_frame("Return")
    df["Year"] = df.index.year
    df["Month"] = df.index.month
    return df.pivot(index="Year", columns="Month", values="Return")

def quarterly_returns(pf: vbt.Portfolio, value: Optional[pd.Series] = None) -> pd.DataFrame:
    """Calculate quarterly return heatmap."""
    value = pf.value() if value is None else value
    ret = value.resample("QE").last().pct_change().dropna()
    if ret.empty:
        return pd.DataFrame()
    df = ret.to_frame("Return")
    df["Year"] = df.index.year
    df["Quarter"] = df.index.quarter
    return df.pivot(index="Year", columns="Quarter", values="Return")