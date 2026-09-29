"""Market data loading shared by backtest and live workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yfinance as yf


class DataLoader:
    """Load and prepare market data."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.root = Path(__file__).resolve().parent.parent
        self.data_dir = self.root / config.get("stock_data_dir", "data/raw/Stock Data")
        self.index_dir = self.root / config.get("index_dir", "data/raw/Indices")
        self.max_ffill_days = config.get("max_ffill_days", 2)

    def resolve_universe(self, universe: Union[str, List[str]]) -> tuple[Optional[str], List[str]]:
        """Resolve a universe name to its ticker list."""
        if isinstance(universe, str):
            path = self.index_dir / f"{universe}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Universe {universe} file not found: {path}")
            df = pd.read_csv(path)
            if "Yahoo" not in df.columns:
                raise ValueError(f"'Yahoo' column not found in {path}")
            tickers = df["Yahoo"].dropna().astype(str).str.strip().unique().tolist()
            return universe, tickers
        if isinstance(universe, (list, tuple, np.ndarray)):
            tickers = list(universe)
            if not tickers:
                raise ValueError("Universe is empty.")
            return None, tickers
        raise TypeError("Universe must be an index name or ticker list.")

    def load(
        self,
        universe: Union[str, List[str]],
        start: str,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load, clean, and forward-fill close prices."""
        universe_name, tickers = self.resolve_universe(universe)
        end_date = pd.Timestamp.today().normalize() if end is None else pd.Timestamp(end)
        data_start = pd.Timestamp(start) - pd.DateOffset(
            years=self.config.get("warmup_years", 2)
        )

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
        else:
            raise ValueError("Data source must be 'pkl' or 'yahoo'.")

        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()
        prices = prices.loc[~prices.index.duplicated()]
        prices = prices.replace([np.inf, -np.inf, 0], np.nan)
        prices = prices.mask(prices <= 0, np.nan)
        prices = prices.ffill(limit=self.max_ffill_days)
        prices = prices.dropna(axis=1, how="all")

        if prices.empty:
            raise ValueError("No valid price data.")
        return prices
