import yaml
import pandas as pd
from pathlib import Path
from src.data_loader import DataLoader
from src.live.selector import StockDecision, MomentumLiveSelector


COLUMN_GROUPS = [
    ["universe", "run_timestamp", "stock", "status", "rank"],
    ["entry_date", "entry_price", "qty_to_buy", "last_price"],
    ["stop_loss_price", "trailing_stop_price"],
    ["mae", "mfe", "highest_price"],
    ["rank_current_month", "roc", "roc_current_month", "roc_history", "vol_90", "momentum_score", "rank_history"],
    ["ma50", "ma200", "extension", "extension2", "macd", "macd_histogram", "accel_6m_4w"],
    ["remarks"],
]


def arrange_live_output(dataframe):
    """Arrange live output fields with blank columns between sections."""
    sections = []
    for index, columns in enumerate(COLUMN_GROUPS):
        sections.append(dataframe[columns])
        if index < len(COLUMN_GROUPS) - 1:
            sections.append(pd.DataFrame({"": pd.NA}, index=dataframe.index))
    return pd.concat(sections, axis=1)

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

output_dir = Path("results")
output_dir.mkdir(parents=True, exist_ok=True)
output_timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
output_path = output_dir / f"live_output_{output_timestamp}.csv"

if live_results:
    live_output = pd.concat(live_results, ignore_index=True)
    arrange_live_output(live_output).to_csv(output_path, index=False)
else:
    arrange_live_output(pd.DataFrame(columns=sum(COLUMN_GROUPS, []))).to_csv(output_path, index=False)

print(f"Live output saved to {output_path}")
