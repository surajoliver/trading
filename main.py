# """Main entry point for backtesting framework."""

# from __future__ import annotations
import argparse
# import logging

# import sys
# from numbers import Number
# from pathlib import Path
# from datetime import datetime
import pandas as pd
import numpy as np

from src.core import DataLoader, run_backtest, calculate_metrics, BacktestResult
from src.strategies import StrategyFactory
# from src.optimization import grid_search, walk_forward_optimization, parameter_sensitivity_analysis
# from src.reporting import generate_html_report, export_excel, generate_pdf_report
from src.utils import setup_logging, load_yaml_config, timer



def add_trade_metrics(close, trade_log):
    """Add MAE and MFE columns to every trade row using the realized path between entry and exit."""
    if trade_log.empty:
        return trade_log

    def first_value(mapping, *keys):
        for key in keys:
            if key in mapping and mapping[key] is not None and not pd.isna(mapping[key]):
                return mapping[key]
        return None

    rows = []
    for _, row in trade_log.iterrows():
        row_dict = row.to_dict()
        symbol = first_value(row_dict, "Column", "Symbol", "Ticker")
        if symbol is None:
            continue

        entry = first_value(row_dict, "Entry_Timestamp", "EntryDate", "Entry Timestamp")
        exit = first_value(row_dict, "Exit_Timestamp", "ExitDate", "Exit Timestamp")
        entry_price = first_value(row_dict, "Entry_Price", "EntryPrice", "Entry Price", "Avg Entry Price")
        exit_price = first_value(row_dict, "Exit_Price", "ExitPrice", "Exit Price", "Avg Exit Price")

        if entry is None or exit is None or entry_price is None or exit_price is None:
            continue

        try:
            entry = pd.Timestamp(entry)
            exit = pd.Timestamp(exit)
            entry_price = float(entry_price)
            exit_price = float(exit_price)
        except (TypeError, ValueError):
            continue

        if symbol not in close.columns:
            continue

        px = close.loc[entry:exit, symbol].dropna()
        if px.empty:
            continue

        mae = max(0.0, 1 - px.min() / entry_price)
        mfe = max(0.0, px.max() / entry_price - 1)

        row_dict["MAE"] = mae
        row_dict["MFE"] = mfe
        rows.append(row_dict)

    if not rows:
        trade_log["MAE"] = np.nan
        trade_log["MFE"] = np.nan
        return trade_log

    trade_log = pd.DataFrame(rows)
    return trade_log


def trade_report(close, trades, strategy, universe, period):
    loss_bins = [-np.inf, -.20, -.10, -.05, 0]
    loss_labels = ["<-20%", "-20--10%", "-10--5%", "-5-0%"]

    profit_bins = [0, .10, .40, 1, np.inf]
    profit_labels = ["0-10%", "10-40%", "40-100%", ">100%"]

    exc_bins = [0, .02, .05, .10, np.inf]
    exc_labels = ["0-2%", "2-5%", "5-10%", ">10%"]

    rows = []

    def first_value(mapping, *keys):
        for key in keys:
            if key in mapping and mapping[key] is not None and not pd.isna(mapping[key]):
                return mapping[key]
        return None

    for _, row in trades.iterrows():
        row_dict = row.to_dict()
        s = (
            first_value(row_dict, "Column", "Symbol", "Ticker")
        )
        if s is None:
            continue
        entry = first_value(row_dict, "Entry_Timestamp", "EntryDate", "Entry Timestamp")
        exit = first_value(row_dict, "Exit_Timestamp", "ExitDate", "Exit Timestamp")
        entry_price = first_value(row_dict, "Entry_Price", "EntryPrice", "Entry Price", "Avg Entry Price")
        exit_price = first_value(row_dict, "Exit_Price", "ExitPrice", "Exit Price", "Avg Exit Price")

        if entry is None or exit is None or entry_price is None or exit_price is None:
            continue

        try:
            entry = pd.Timestamp(entry)
            exit = pd.Timestamp(exit)
            entry_price = float(entry_price)
            exit_price = float(exit_price)
        except (TypeError, ValueError):
            continue

        if s not in close.columns:
            continue

        pnl = exit_price / entry_price - 1
        mae = first_value(row_dict, "MAE")
        mfe = first_value(row_dict, "MFE")
        if mae is None or mfe is None:
            px = close.loc[entry:exit, s].dropna()
            if px.empty:
                continue
            mae = max(0, 1 - px.min() / entry_price)
            mfe = max(0, px.max() / entry_price - 1)
        typ = "Win" if pnl > 0 else "Loss"

        rows.append({
            "type": typ,
            "pnl": pnl,
            "mae": mae,
            "mfe": mfe,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame()

    out = []

    summary_row = {
        "strategy": strategy,
        "universe": universe,
        "period": period,
        "type": "Summary",
        "total_trades": int(len(df)),
    }
    out.append(summary_row)

    # P&L distribution across all trades in one monotonic binning.
    pnl_bins = [-np.inf, -0.20, -0.10, -0.05, 0.0, 0.10, 0.40, 1.0, np.inf]
    pnl_labels = ["<-20%", "-20--10%", "-10--5%", "-5-0%", "0-10%", "10-40%", "40-100%", ">100%"]

    for typ, g in df.groupby("type"):
        row = {
            "strategy": strategy,
            "universe": universe,
            "period": period,
            "type": typ,
            "total_trades": int(len(g)),
        }

        pnl = g["pnl"]
        if typ == "Win":
            row["max"] = float(pnl.max())
            row["min"] = float(pnl.min())
            row["avg"] = float(pnl.mean())
            row["median"] = float(pnl.median())
        else:
            row["max"] = float(pnl.min())
            row["min"] = float(pnl.max())
            row["avg"] = float(pnl.mean())
            row["median"] = float(pnl.median())

        row.update({
            f"MAE {k}": v
            for k, v in pd.cut(g["mae"], exc_bins, labels=exc_labels, right=False).value_counts().items()
        })
        row.update({
            f"MFE {k}": v
            for k, v in pd.cut(g["mfe"], exc_bins, labels=exc_labels, right=False).value_counts().items()
        })

        row.update({
            f"{'Loss ' + str(k) if k.startswith('-') else 'Profit ' + str(k)}": v
            for k, v in pd.cut(pnl, bins=pnl_bins, labels=pnl_labels, right=False).value_counts().items()
        })

        out.append(row)

    df_out = pd.DataFrame(out)
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns
    df_out[numeric_cols] = df_out[numeric_cols].round(2)
    return df_out


def run_single_experiment(cfg, universe, start, end, strat_cfg, close):
    name = strat_cfg["name"]

    strategy = StrategyFactory.create(
        name, **strat_cfg.get("params", {})
    )
    strategy.prepare(close)

    weights = strategy.generate_weights(
        close=close,
        max_positions=cfg["backtest"].get("max_positions", 10),
        start=start,
    )

    result = run_backtest(
        close=close,
        target_weights=weights,
        config=cfg["backtest"],
        strategy_name=name,
    )

    metrics = result.metrics or calculate_metrics(
        result.portfolio, value=result.value
    )

    trade_log = result.trade_log.copy()

    if not trade_log.empty:
        trade_log = add_trade_metrics(close, trade_log)
        trade_log["strategy"] = name
        trade_log["universe"] = universe
        trade_log["start_date"] = start
        trade_log["end_date"] = end

    df_trade_report = trade_report(close, trade_log, name, universe, start)


    return {
        "Universe": universe,
        "Period": start,
        "Strategy": name,
        **metrics,
    }, trade_log, df_trade_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--universe")
    p.add_argument("--strategies", nargs="+")
    return p.parse_args()


@timer
def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)

    logger = setup_logging(cfg.get("logging", {}))
    logger.info("BACKTEST FRAMEWORK STARTED")

    bt = cfg["backtest"]
    universes = [args.universe] if args.universe else bt["universes"]

    strategies = [
        s for s in cfg.get("strategies", [])
        if s.get("enabled", True)
        and (not args.strategies or s["name"] in args.strategies)
    ]

    loader = DataLoader(cfg.get("data", {}))
    results, logs, reports = [], [], []

    for universe in universes:
        for period in bt["periods"]:
            start, end = period["start"], period["end"]
            close = loader.load(universe, start, end)

            for strat_cfg in strategies:
                result, log, report = run_single_experiment(
                    cfg, universe, start, end, strat_cfg, close
                )
                results.append(result)

                if not log.empty:
                    logs.append(log)

                if not report.empty:
                    reports.append(report)

    summary = pd.DataFrame(results).rename(
        columns={"Annualized Return": "Cagr%"}
    ).round(2)

    trade_log = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()

    trade_report = pd.concat(reports, ignore_index=True) if reports else pd.DataFrame()

    summary = summary.round(2)
    trade_log = trade_log.round(2)
    summary.to_csv("summary.csv", index=False)
    trade_log.to_csv("trade_log.csv", index=False)
    trade_report.to_csv("trade_report.csv", index=False)

    print(summary.to_string(index=False))

    return summary, trade_log


if __name__ == "__main__":
    main()


# def format_console_value(value):
#     """Format numeric console values to a maximum of two decimals."""
#     if isinstance(value, Number) and not isinstance(value, (int, np.integer)):
#         return f"{value:.2f}"
#     return str(value)

# def run_single_experiment(cfg, universe, start, end, strategy, close):





#         result = run_backtest(
#             close=close,
#             target_weights=weights,
#             config=cfg["backtest"],
#             strategy_name=name,
#         )
#         metric_result = process_result(
#             result,
#             strat_cfg["name"],
#             universe,
#             start,
#         )
#         trade_log_map[name] = result.trade_log
#         results.append(metric_result)
#     return results, trade_log_map[name] = result.trade_log

# def parse_args():
#     p = argparse.ArgumentParser(description="Backtest trading strategies")
#     p.add_argument("--config", default="config.yaml")
#     p.add_argument("--universe")
#     p.add_argument("--strategies", nargs="+")
#     p.add_argument("--optimize", action="store_true")
#     p.add_argument("--walk-forward", action="store_true")
#     return p.parse_args()


# def setup_logging(cfg):
#     logging.basicConfig(
#         level=getattr(logging, cfg.get("level", "INFO").upper()),
#         format=cfg.get("format", "%(asctime)s | %(levelname)s | %(message)s"),
#     )
#     return logging.getLogger(__name__)


# def process_result(result, strategy, universe, period):
#     metrics = result.metrics or calculate_metrics(
#         result.portfolio,
#         value=result.value
#     )

#     return {
#         "Universe": universe,
#         "Period": period,
#         "Strategy": strategy,
#         **metrics,
#     }

# @timer
# def main():
#     args = parse_args()
#     cfg = load_yaml_config(args.config)
#     logger = setup_logging(cfg.get("logging", {}))
#     logger.info("BACKTEST FRAMEWORK STARTED")
#     logger.info("=" * 80)

#     backtest_cfg = cfg["backtest"]
#     universes = backtest_cfg["universes"]
#     periods = backtest_cfg["periods"]
#     strat_configs = cfg.get("strategies", [])

#     if args.strategies: 
#         strat_configs = [ 
#             strat_cfg for strat_cfg in strat_configs 
#             if strat_cfg.get("name") in args.strategies 
#         ]

#         enabled_strategies = [ 
#             strat_cfg for strat_cfg in strat_configs 
#             if strat_cfg.get("enabled", True) 
#         ]
#         data_cfg = cfg.get("data", {})
#         loader = DataLoader(data_cfg)

#         params = strat_cfg.get("params", {})

#         strategy = StrategyFactory.create(name, **params)
#         strategy.prepare(close)

#         weights = strategy.generate_weights(
#             close=close,
#             max_positions=cfg["backtest"].get("max_positions", 10),
#         )


#     final_results = []
#     trade_logs = []
#     for universe in universes:
#         for period in periods:
#             start,end=period["start"],period["end"]
#             data_cfg = cfg.get("data", {})
#             loader = DataLoader(data_cfg)
#             close = loader.load(universe=universe, start=start, end=end)
#             for name, strategy in enabled_strategies:
#                 result = run_single_experiment(
#                     cfg=cfg,
#                     universe=universe,
#                     start=start,
#                     end=end,
#                     strategy=name,
#                     close=close
#                 )
#                 df = result.trade_log.copy()

#                 df["strategy"] = name
#                 df["universe"] = universe
#                 df["start_date"] = period["start"]

#                 trade_logs.append(df)

#                 # final_results = final_results + results
                
#     trade_log = pd.concat(trade_logs, ignore_index=True)



#     # for result in final_results:
#     #     print(result)
#     summary = pd.DataFrame(final_results)
#     summary = summary.round(2)
#     summary = summary.rename(columns={"Annualized Return": "Cagr%"})
#     # summary["Cagr%"] *= 100
#     # summary["Win Rate"] *= 100
    
#     pd.set_option("display.float_format", "{:.2f}".format)
#     summary.to_csv('summary.csv')
#     summary = summary[["Universe", "Period", "Strategy", "Sharpe", "Calmar", "Cagr%","Max Drawdown","Ulcer Index", "Profit Factor", "Win Rate",  "Total Trades"]]

#     print(summary.to_string(float_format="%.2f"))

    
            
#     #         # Get param grid
#     #         param_grid = opt_cfg.get("grid", {}).get(name, {})
#     #         if not param_grid:
#     #             logger.warning(f"No parameter grid for {name}, skipping")
#     #             continue
            
#     #         # Run grid search
#     #         opt_results = grid_search(
#     #             strategy_builder=name,
#     #             close=close,
#     #             base_params=strat_cfg.get("params", {}),
#     #             param_grid=param_grid,
#     #             run_config=backtest_cfg,
#     #             metric=opt_cfg.get("metric", "Sharpe"),
#     #             n_jobs=opt_cfg.get("n_jobs", -1),
#     #         )
            
#     #         # Save optimization results
#     #         output_dir = Path(cfg.get("reporting", {}).get("output_dir", "results"))
#     #         output_dir.mkdir(parents=True, exist_ok=True)
#     #         opt_results.to_csv(
#     #             output_dir / f"{backtest_cfg.get('universe')}_{name}_optimization.csv",
#     #             index=False,
#     #             float_format="%.2f",
#     #         )
            
#     #         # Print best parameters
#     #         if not opt_results.empty:
#     #             best = opt_results.iloc[0]
#     #             print(f"\nBest parameters for {name}:")
#     #             for key in param_grid.keys():
#     #                 print(f"  {key}: {format_console_value(best[key])}")
#     #             metric_name = opt_cfg.get('metric', 'Sharpe')
#     #             print(f"Best {metric_name}: {format_console_value(best[metric_name])}")
        
#     #     return
    
#     # # Walk-forward optimization
#     # if args.walk_forward:
#     #     logger.info("Running walk-forward optimization...")
#     #     opt_cfg = cfg.get("optimization", {})
        
#     #     for strat_cfg in enabled_strategies:
#     #         name = strat_cfg["name"]
#     #         param_grid = opt_cfg.get("grid", {}).get(name, {})
#     #         if not param_grid:
#     #             continue
            
#     #         wf_results = walk_forward_optimization(
#     #             close=close,
#     #             strategy_name=name,
#     #             base_params=strat_cfg.get("params", {}),
#     #             param_grid=param_grid,
#     #             run_config=backtest_cfg,
#     #             train_years=opt_cfg.get("train_years", 3),
#     #             test_months=opt_cfg.get("test_months", 6),
#     #             metric=opt_cfg.get("metric", "Sharpe"),
#     #             n_jobs=opt_cfg.get("n_jobs", -1),
#     #         )
            
#     #         output_dir = Path(cfg.get("reporting", {}).get("output_dir", "results"))
#     #         output_dir.mkdir(parents=True, exist_ok=True)
#     #         wf_results.to_csv(
#     #             output_dir / f"{backtest_cfg.get('universe')}_{name}_walk_forward.csv",
#     #             index=False,
#     #             float_format="%.2f",
#     #         )
#     #         print(f"\nWalk-forward results saved for {name}")
        
#     #     return
    
#     # # Normal backtest mode
#     # logger.info("Running backtests...")
    
#     # for strat_cfg in enabled_strategies:
#     #     name = strat_cfg["name"]
#     #     params = strat_cfg.get("params", {})
        
#     #     logger.info(f"Running strategy: {name}")
        
#     #     # Create strategy
#     #     strategy = StrategyFactory.create(name, **params)
#     #     strategy.prepare(close)
        
#     #     # Generate weights with risk management
#     #     risk_cfg = cfg.get("risk", {})
#     #     weights = strategy.generate_weights(
#     #         close=close,
#     #         max_positions=backtest_cfg.get("max_positions", 10),
#     #         start=backtest_cfg.get("start"),
#     #         sizing=risk_cfg.get("position_sizing", "equal"),
#     #         vol_lookback=risk_cfg.get("volatility_lookback", 21),
#     #         stop_loss_pct=risk_cfg.get("stop_loss_pct"),
#     #         take_profit_pct=risk_cfg.get("take_profit_pct"),
#     #     )
#     #     weights.to_csv(
#     #         f"weight_{backtest_cfg.get('universe')}_{name}.csv",
#     #         float_format="%.2f",
#     #     )
        
#     #     # Run backtest
#     #     result = run_backtest(
#     #         close=close,
#     #         target_weights=weights,
#     #         config=backtest_cfg,
#     #         strategy_name=name,
#     #     )
        



# if __name__ == "__main__":
#     main()