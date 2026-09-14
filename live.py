import yaml
from src.core import DataLoader
from src.live.selector import StockDecision, MomentumLiveSelector

with open("live.yaml", "r") as f:
    cfg = yaml.safe_load(f)

portfolio = cfg["portfolios"]
universes = list(portfolio.keys())
data_cfg = cfg.get("data", {})
loader = DataLoader(data_cfg)

for universe, portfolio in cfg["portfolios"].items():
    max_positions = portfolio["max_positions"]
    sell_rank = portfolio["sell_rank"]
    skip_months = portfolio["skip_months"]
    buy_freq = portfolio["buy_freq"]
    sell_freq = portfolio["sell_freq"]
    close = loader.load(universe=universe, start="2026-01-01")
    print(f"\n{universe:~^50}")
    momentum = MomentumLiveSelector(skip=skip_months*21, sell_rank=sell_rank, max_positions=max_positions)
    momentum.prepare(close)
    holdings = list(portfolio["holdings"].keys())
    stock_decision = momentum.select(close, holdings=holdings)
    print(stock_decision)
