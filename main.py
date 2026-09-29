"""Run configured backtests and save summary outputs."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyfiglet, random

from src.backtest.core import calculate_metrics, run_backtest
from src.data_loader import DataLoader
from src.strategies import StrategyFactory
from src.utils import load_yaml_config, setup_logging, timer


CSV_COLS = [
    "Universe", "Period", "Strategy",
    "Total Return", "CAGR%", "Expectancy",
    "Sharpe", "Sortino", "Calmar", "Gain-to-Pain Ratio",
    "Max Drawdown", "Max Drawdown Duration (days)", "Ulcer Index", "Risk of Ruin",
    "Total Trades", "Trade Frequency / Year",
    "Win Rate", "Profit Factor", "Average Win", "Average Loss",
]


CONSOLE_COLS = ["Universe", "Period", "Strategy", "CAGR%", "MaxDD%", "Calmar", "Trades", "PF"]

TRADE_LOG_COLS = [
    "Exit Trade Id", "Position Id", "strategy", "universe", "Stock",
    "Direction", "EntryDate", "Avg Entry Price", "Size", "Entry Fees",
    "ExitDate", "Avg Exit Price", "Exit Fees", "Status",
    "PnL", "Return", "MAE", "MFE", "start_date", "end_date",
]

TRADE_REPORT_COLS = [
    "strategy", "universe", "period", "type", "total_trades", "min", "max", "avg", "median",
    "MAE >10%", "MAE 5-10%", "MAE 2-5%", "MAE 0-2%",
    "MFE >10%", "MFE 5-10%", "MFE 2-5%", "MFE 0-2%",
    "Loss <-20%", "Loss -20--10%", "Loss -10--5%", "Loss -5-0%",
    "Profit 0-10%", "Profit 10-40%", "Profit 40-100%", "Profit >100%",
]

quotes = [
    "Success is built one disciplined day at a time.",
    "Discipline turns intention into achievement.",
    "Consistency beats intensity when intensity fades.",
    "Do the work, even when nobody is watching.",
    "Small actions repeated daily create extraordinary results.",
    "Focus on progress, not perfection.",
    "Your habits quietly shape your future.",
    "Discipline is freedom in disguise.",
    "Success rewards those who stay in the game.",
    "Patience is a powerful form of discipline.",

    "Keep your standards high and your excuses low.",
    "Master yourself before trying to master anything else.",
    "Confidence comes from keeping promises to yourself.",
    "Calm people command attention.",
    "Speak less, listen more, observe everything.",
    "Presence begins with being fully present.",
    "Walk with purpose and speak with clarity.",
    "Confidence doesn't need to announce itself.",
    "A calm mind creates a strong presence.",
    "Charisma begins with genuine interest in others.",

    "Make people feel heard and they will remember you.",
    "Listen carefully; people reveal what matters to them.",
    "Good leaders create confidence in others.",
    "Respect is earned through consistency.",
    "Strong character speaks louder than strong words.",
    "Be decisive, but remain open to learning.",
    "Your attitude enters the room before you do.",
    "Confidence is quiet; insecurity is noisy.",
    "Stand tall, think clearly, speak deliberately.",
    "Control your reactions and you control your presence.",

    "Don't chase attention; become worth noticing.",
    "Be interested, not impressive.",
    "The strongest person in the room may be the calmest.",
    "Clarity creates confidence.",
    "Courage grows every time you act despite uncertainty.",
    "You don't need permission to improve yourself.",
    "The quality of your decisions shapes the quality of your life.",
    "Learn from yesterday, execute today, prepare for tomorrow.",
    "Protect your energy for what truly matters.",
    "Become the person your goals require.",

    "When the plan is right, patience becomes an advantage.",
    "Discipline is doing what needs to be done without negotiation.",
    "Don't let temporary emotions make permanent decisions.",
    "Stay humble when you win and composed when you lose.",
    "Your reputation is built through repeated behavior.",
    "Excellence is ordinary actions performed exceptionally well.",
    "Control what you can; accept what you cannot.",
    "The ability to stay calm is a competitive advantage.",
    "Keep learning, keep adapting, keep moving.",
    "Build quietly. Let the results speak."
]


def _first_value(mapping, *keys):
    for key in keys:
        value = mapping.get(key)
        if value is not None and not pd.isna(value):
            return value
    return None


def add_trade_metrics(close: pd.DataFrame, trade_log: pd.DataFrame) -> pd.DataFrame:
    """Add MAE and MFE calculated over each trade's realized price path."""
    if trade_log.empty:
        return trade_log

    rows = []
    for _, row in trade_log.iterrows():
        values = row.to_dict()
        symbol = _first_value(values, "Column", "Symbol", "Ticker")
        entry = _first_value(values, "Entry_Timestamp", "EntryDate", "Entry Timestamp")
        exit_date = _first_value(values, "Exit_Timestamp", "ExitDate", "Exit Timestamp")
        entry_price = _first_value(values, "Entry_Price", "EntryPrice", "Entry Price", "Avg Entry Price")
        exit_price = _first_value(values, "Exit_Price", "ExitPrice", "Exit Price", "Avg Exit Price")

        if None in (symbol, entry, exit_date, entry_price, exit_price) or symbol not in close:
            continue

        try:
            entry = pd.Timestamp(entry)
            exit_date = pd.Timestamp(exit_date)
            entry_price = float(entry_price)
            float(exit_price)
        except (TypeError, ValueError):
            continue

        prices = close.loc[entry:exit_date, symbol].dropna()
        if prices.empty or entry_price <= 0:
            continue

        values["MAE"] = max(0.0, 1 - prices.min() / entry_price)
        values["MFE"] = max(0.0, prices.max() / entry_price - 1)
        rows.append(values)

    if not rows:
        trade_log = trade_log.copy()
        trade_log["MAE"] = np.nan
        trade_log["MFE"] = np.nan
        return trade_log
    return pd.DataFrame(rows)


def trade_report(
    close: pd.DataFrame,
    trades: pd.DataFrame,
    strategy: str,
    universe: str,
    period: str,
) -> pd.DataFrame:
    """Summarize P&L, MAE, and MFE distributions by trade result."""
    if trades.empty:
        return pd.DataFrame(columns=TRADE_REPORT_COLS)

    rows = []
    for _, row in trades.iterrows():
        values = row.to_dict()
        symbol = _first_value(values, "Column", "Symbol", "Ticker")
        entry_price = _first_value(values, "Entry_Price", "EntryPrice", "Entry Price", "Avg Entry Price")
        exit_price = _first_value(values, "Exit_Price", "ExitPrice", "Exit Price", "Avg Exit Price")
        if symbol is None or entry_price is None or exit_price is None:
            continue

        try:
            entry_price = float(entry_price)
            exit_price = float(exit_price)
        except (TypeError, ValueError):
            continue
        if entry_price <= 0:
            continue

        mae = _first_value(values, "MAE")
        mfe = _first_value(values, "MFE")
        if mae is None or mfe is None:
            entry = _first_value(values, "Entry_Timestamp", "EntryDate", "Entry Timestamp")
            exit_date = _first_value(values, "Exit_Timestamp", "ExitDate", "Exit Timestamp")
            if entry is None or exit_date is None or symbol not in close:
                continue
            prices = close.loc[pd.Timestamp(entry):pd.Timestamp(exit_date), symbol].dropna()
            if prices.empty:
                continue
            mae = max(0.0, 1 - prices.min() / entry_price)
            mfe = max(0.0, prices.max() / entry_price - 1)

        pnl = exit_price / entry_price - 1
        rows.append({"type": "Win" if pnl > 0 else "Loss", "pnl": pnl, "mae": mae, "mfe": mfe})

    if not rows:
        return pd.DataFrame(columns=TRADE_REPORT_COLS)

    trades_df = pd.DataFrame(rows)
    output = [{
        "strategy": strategy,
        "universe": universe,
        "period": period,
        "type": "Summary",
        "total_trades": len(trades_df),
    }]
    pnl_bins = [-np.inf, -0.20, -0.10, -0.05, 0.0, 0.10, 0.40, 1.0, np.inf]
    pnl_labels = ["<-20%", "-20--10%", "-10--5%", "-5-0%", "0-10%", "10-40%", "40-100%", ">100%"]
    excursion_bins = [0, 0.02, 0.05, 0.10, np.inf]
    excursion_labels = ["0-2%", "2-5%", "5-10%", ">10%"]

    for result_type, group in trades_df.groupby("type"):
        pnl = group["pnl"]
        row = {
            "strategy": strategy,
            "universe": universe,
            "period": period,
            "type": result_type,
            "total_trades": len(group),
            "max": float(pnl.max() if result_type == "Win" else pnl.min()),
            "min": float(pnl.min() if result_type == "Win" else pnl.max()),
            "avg": float(pnl.mean()),
            "median": float(pnl.median()),
        }
        for column, prefix in (("mae", "MAE"), ("mfe", "MFE")):
            row.update({
                f"{prefix} {label}": count
                for label, count in pd.cut(
                    group[column], excursion_bins, labels=excursion_labels, right=False
                ).value_counts().items()
            })
        row.update({
            f"{'Loss' if label.startswith('-') else 'Profit'} {label}": count
            for label, count in pd.cut(
                pnl, pnl_bins, labels=pnl_labels, right=False
            ).value_counts().items()
        })
        output.append(row)

    result = pd.DataFrame(output)
    numeric_columns = result.select_dtypes(include=[np.number]).columns
    result[numeric_columns] = result[numeric_columns].round(2)
    return result.reindex(columns=TRADE_REPORT_COLS)


def run_single_experiment(cfg, universe, period, strategy_config, close):
    start, end = period["start"], period["end"]
    name = strategy_config["name"]
    params = strategy_config.get("params", {}).copy()
    sell_rank_by_universe = cfg["backtest"].get("sell_rank_by_universe", {})
    if name.startswith("momentum") and universe in sell_rank_by_universe:
        params["sell_rank"] = sell_rank_by_universe[universe]

    strategy = StrategyFactory.create(name, **params)
    strategy.prepare(close)
    weights = strategy.generate_weights(
        close=close,
        max_positions=cfg["backtest"].get("max_positions", 10),
        start=start,
    )
    result = run_backtest(close, weights, cfg["backtest"], name)
    metrics = result.metrics or calculate_metrics(result.portfolio, value=result.value)
    trade_log = add_trade_metrics(close, result.trade_log.copy())

    if not trade_log.empty:
        if "Column" in trade_log.columns:
            trade_log["Stock"] = trade_log["Column"]
        trade_log["strategy"] = name
        trade_log["universe"] = universe
        trade_log["start_date"] = start
        trade_log["end_date"] = end

    return {
        "Universe": universe,
        "Period": start,
        "Strategy": name,
        **metrics,
    }, trade_log, trade_report(close, trade_log, name, universe, start)


def parse_args():
    parser = argparse.ArgumentParser(description="Run configured backtests")
    parser.add_argument("--config", default="backtest.yaml")
    parser.add_argument("--universe")
    parser.add_argument("--strategies", nargs="+")
    return parser.parse_args()


def save_outputs(config, summary, trade_log, trade_report):
    output_dir = Path(config.get("reporting", {}).get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary.reindex(columns=CSV_COLS).to_csv(output_dir / f"summary_{timestamp}.csv", index=False)
    trade_log.reindex(columns=TRADE_LOG_COLS).to_csv(
        output_dir / f"trade_log_{timestamp}.csv", index=False
    )
    trade_report.reindex(columns=TRADE_REPORT_COLS).to_csv(
        output_dir / f"trade_report_{timestamp}.csv", index=False
    )


@timer
def main():
    quote = random.choice(quotes)
    # print(pyfiglet.figlet_format(quote, font="small"))
    print(quote)
    args = parse_args()
    config = load_yaml_config(args.config)
    setup_logging(config.get("logging", {}))
    backtest_config = config["backtest"]
    universes = [args.universe] if args.universe else backtest_config["universes"]
    strategies = [
        item for item in config.get("strategies", [])
        if item.get("enabled", True)
        and (not args.strategies or item["name"] in args.strategies)
    ]
    loader = DataLoader(config.get("data", {}))
    results, logs, reports = [], [], []

    for universe in universes:
        for period in backtest_config["periods"]:
            close = loader.load(universe, period["start"], period["end"])
            for strategy_config in strategies:
                result, trade_log, report = run_single_experiment(
                    config, universe, period, strategy_config, close
                )
                results.append(result)
                if not trade_log.empty:
                    logs.append(trade_log)
                if not report.empty:
                    reports.append(report)

    summary = pd.DataFrame(results).rename(columns={"Annualized Return": "CAGR%"})
    summary = summary.reindex(columns=CSV_COLS).round(2)
    trade_log = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    if not trade_log.empty:
        numeric_columns = trade_log.select_dtypes(include=[np.number]).columns
        trade_log[numeric_columns] = trade_log[numeric_columns].round(2)
    trade_report_df = pd.concat(reports, ignore_index=True) if reports else pd.DataFrame()
    save_outputs(config, summary, trade_log, trade_report_df)

    console_summary = pd.DataFrame({
        "Universe": summary["Universe"],
        "Period": summary["Period"],
        "Strategy": summary["Strategy"],
        "CAGR%": summary["CAGR%"] * 100,
        "MaxDD%": summary["Max Drawdown"] * 100,
        "Calmar": summary["Calmar"],
        "Trades": summary["Total Trades"],
        "PF": summary["Profit Factor"],
    }).round(2)
    print(console_summary[CONSOLE_COLS].to_string(index=False))
    return summary, trade_log


if __name__ == "__main__":
    main()
