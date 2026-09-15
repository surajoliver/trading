import yaml
import pandas as pd
from src.core import DataLoader
from src.live.selector import StockDecision, MomentumLiveSelector

with open("live.yaml", "r") as f:
    cfg = yaml.safe_load(f)

portfolio = cfg["portfolios"]
universes = list(portfolio.keys())
data_cfg = cfg.get("data", {})
loader = DataLoader(data_cfg)
live_results = []
run_timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
stop_loss_pct = cfg.get("risk", {}).get("stop_loss_pct", 0.20)

for universe, portfolio in cfg["portfolios"].items():
    max_positions = portfolio["max_positions"]
    sell_rank = portfolio["sell_rank"]
    skip_months = portfolio["skip_months"]
    close = loader.load(universe=universe, start="2026-01-01")
    print(f"\n{universe:~^50}")
    momentum = MomentumLiveSelector(skip=skip_months*21, sell_rank=sell_rank, max_positions=max_positions)
    momentum.prepare(close)
    holding_details = portfolio["holdings"]
    holdings = list(holding_details.keys())
    entry_prices = {
        symbol: details.get("buy_price")
        for symbol, details in holding_details.items()
    }
    entry_dates = {
        symbol: details.get("buy_date")
        for symbol, details in holding_details.items()
    }
    stock_decision = momentum.select(
        close,
        holdings=holdings,
        entry_prices=entry_prices,
        entry_dates=entry_dates,
        stop_loss_pct=stop_loss_pct,
    )
    print(stock_decision)

    decision_df = stock_decision.to_dataframe()
    if not decision_df.empty:
        decision_df.insert(0, "universe", universe)
        decision_df.insert(1, "run_timestamp", run_timestamp)
        live_results.append(decision_df)

if live_results:
    pd.concat(live_results, ignore_index=True).to_csv("live_output.csv", index=False)
else:
    pd.DataFrame().to_csv("live_output.csv", index=False)

print("Live output saved to live_output.csv")
