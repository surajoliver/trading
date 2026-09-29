"""VectorBT backtest engine and performance metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import vectorbt as vbt


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
    config: Dict,
    strategy_name: str = "strategy",
) -> BacktestResult:
    """Run a vectorbt backtest with target portfolio weights."""
    close, target_weights = close.align(target_weights, join="left", axis=0)
    target_weights = target_weights.reindex(columns=close.columns, fill_value=0)

    portfolio = vbt.Portfolio.from_orders(
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
    value = portfolio.value()
    metrics = calculate_metrics(portfolio, value=value)
    diagnostics = (
        build_diagnostics(close, target_weights, value)
        if config.get("build_diagnostics", False)
        else pd.DataFrame()
    )

    return BacktestResult(
        strategy=strategy_name,
        portfolio=portfolio,
        target_weights=target_weights,
        diagnostics=diagnostics,
        trade_log=get_trade_log(portfolio),
        value=value,
        metrics=metrics,
    )


def build_diagnostics(close: pd.DataFrame, weights: pd.DataFrame, value: pd.Series) -> pd.DataFrame:
    rows = weights.stack().rename("TargetWeight").reset_index()
    rows.columns = ["Date", "Symbol", "TargetWeight"]
    rows["Action"] = np.where(rows["TargetWeight"].eq(0), "SELL", "BUY")
    rows["Price"] = [
        close.at[dt, symbol]
        for dt, symbol in zip(rows["Date"], rows["Symbol"])
    ]
    rows["PortfolioValue"] = rows["Date"].map(value)
    return rows[["Date", "Symbol", "Action", "TargetWeight", "Price", "PortfolioValue"]]


def get_trade_log(portfolio: vbt.Portfolio) -> pd.DataFrame:
    """Extract a normalized trade log from a vectorbt portfolio."""
    try:
        trades = portfolio.trades.records_readable.copy()
        if trades.empty:
            return trades
        return trades.rename(
            columns={
                "Entry Timestamp": "EntryDate",
                "Exit Timestamp": "ExitDate",
                "Entry Price": "EntryPrice",
                "Exit Price": "ExitPrice",
            }
        )
    except Exception:
        return pd.DataFrame()


def calculate_metrics(
    portfolio: vbt.Portfolio,
    benchmark: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    value: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """Calculate return, risk, drawdown, and trade metrics."""
    value = portfolio.value() if value is None else value
    if len(value) < 2:
        return {}

    ret = value.pct_change().dropna()
    if ret.empty:
        return {}

    total_return = value.iloc[-1] / value.iloc[0] - 1
    years = (value.index[-1] - value.index[0]).days / 365.25
    annualized = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan

    ret_mean = ret.mean()
    ret_std = ret.std()
    sharpe = (ret_mean - risk_free_rate) / ret_std * np.sqrt(252) if ret_std > 0 else np.nan
    downside = ret[ret < 0].std()
    sortino = (ret_mean - risk_free_rate) / downside * np.sqrt(252) if downside and downside > 0 else np.nan

    peak = value.cummax()
    drawdown = (value - peak) / peak
    max_dd = abs(drawdown.min()) if not drawdown.empty else np.nan
    max_dd_duration = 0
    if not drawdown.empty and drawdown.min() < 0:
        trough_date = drawdown.idxmin()
        peak_date = value.cummax().loc[:trough_date].idxmax()
        max_dd_duration = (trough_date - peak_date).days

    calmar = annualized / max_dd if max_dd and max_dd > 0 else np.nan
    ulcer = np.sqrt((drawdown ** 2).mean()) if not drawdown.empty else np.nan
    gains = ret[ret > 0].sum()
    losses = abs(ret[ret < 0].sum())
    gain_pain = gains / losses if losses != 0 else np.nan

    trades = portfolio.trades.records_readable
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
        ratio = abs(avg_win / avg_loss) if pd.notna(avg_loss) and avg_loss != 0 else np.nan
        risk_ruin = ((1 - win_rate) / (1 + win_rate * ratio)) ** 100 if pd.notna(ratio) and ratio > 0 else np.nan
    else:
        win_rate = profit_factor = avg_win = avg_loss = expectancy = trade_freq = risk_ruin = np.nan
        total_trades = 0

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
        "Win Rate": win_rate,
        "Profit Factor": profit_factor,
        "Average Win": avg_win,
        "Average Loss": avg_loss,
        "Expectancy": expectancy,
        "Total Trades": total_trades,
        "Trade Frequency / Year": trade_freq,
        "Risk of Ruin": risk_ruin,
    }

    if benchmark is not None:
        benchmark = benchmark.reindex(value.index).ffill().dropna()
        if len(benchmark) > 1:
            benchmark_return = benchmark.iloc[-1] / benchmark.iloc[0] - 1
            metrics["Benchmark Return"] = benchmark_return
            metrics["Excess Return"] = total_return - benchmark_return

    return metrics


def monthly_returns(portfolio: vbt.Portfolio, value: Optional[pd.Series] = None) -> pd.DataFrame:
    value = portfolio.value() if value is None else value
    returns = value.resample("ME").last().pct_change().dropna()
    if returns.empty:
        return pd.DataFrame()
    frame = returns.to_frame("Return")
    frame["Year"] = frame.index.year
    frame["Month"] = frame.index.month
    return frame.pivot(index="Year", columns="Month", values="Return")


def quarterly_returns(portfolio: vbt.Portfolio, value: Optional[pd.Series] = None) -> pd.DataFrame:
    value = portfolio.value() if value is None else value
    returns = value.resample("QE").last().pct_change().dropna()
    if returns.empty:
        return pd.DataFrame()
    frame = returns.to_frame("Return")
    frame["Year"] = frame.index.year
    frame["Quarter"] = frame.index.quarter
    return frame.pivot(index="Year", columns="Quarter", values="Return")
