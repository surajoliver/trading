"""Trading strategy implementations."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Set, Optional
import numpy as np
import pandas as pd
from src.utils import get_trading_dates

from abc import ABC, abstractmethod
from typing import Optional, Set, List, Dict, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Order:
    """Represents a single trade order."""
    date: pd.Timestamp
    symbol: str
    action: str  # 'BUY' or 'SELL'
    price: float
    weight: float
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date,
            'symbol': self.symbol,
            'action': self.action,
            'price': self.price,
            'weight': self.weight,
            'reason': self.reason
        }

class Strategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, max_positions: int = 10, buy_freq: str = "daily", sell_freq: str = "daily", seed: int = 42, **kwargs):
        self.max_positions = max_positions
        self.buy_freq, self.sell_freq = buy_freq, sell_freq
        self.seed = seed
        self._orders: List[Order] = []
        self.buy_dates: Set[pd.Timestamp] = set()
        self.sell_dates: Set[pd.Timestamp] = set()

    @abstractmethod
    def prepare(self, prices: pd.DataFrame): pass
    
    @abstractmethod
    def get_buys(self, prices: pd.DataFrame, dt: pd.Timestamp, holdings: Set[str], needed: int) -> List[str]: pass
    
    @abstractmethod
    def get_sells(self, prices: pd.DataFrame, dt: pd.Timestamp, holdings: Set[str]) -> List[str]: pass
 
    def _get_trading_dates(self, index: pd.DatetimeIndex, freq: str) -> Set[pd.Timestamp]:
        """Get trading dates based on frequency."""
        valid_rules = {'D', 'W', 'M', 'Q', 'Y', 'A'}
        rule =freq.upper()
        if rule in valid_rules:
            if freq == 'D':
                return set(index)        
            return set(
                pd.Series(index=index)
                .groupby(index.to_period(rule))
                .head(1)
                .index
            )
        return set(index)

    def _get_buy_dates(self, index: pd.DatetimeIndex) -> Set[pd.Timestamp]:
        """Get buy dates based on buy_freq."""
        return self._get_trading_dates(index, self.buy_freq)
    
    def _get_sell_dates(self, index: pd.DatetimeIndex) -> Set[pd.Timestamp]:
        """Get sell dates based on sell_freq."""
        return self._get_trading_dates(index, self.sell_freq)
    
    def generate_weights(self, close: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate weights and track all orders."""
        self._orders = []

        start = kwargs.get("start")
        if not start is None:
            close = close.loc[start:]
        
        # Get buy and sell dates
        self.buy_dates = self._get_buy_dates(close.index)
        self.sell_dates = self._get_sell_dates(close.index)
        
        # Initialize
        #weights = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        weights_array = np.full(
            (len(close.index), len(close.columns)),
            np.nan,
            dtype=np.float64
        )
        col_map = {sym: i for i, sym in enumerate(close.columns)}
        
        holdings = set()
        weight_per_pos = 1.0 / self.max_positions
        
        # Track date categories for debugging
        date_categories = {}

        trade_dates = sorted(self.buy_dates | self.sell_dates)
        for dt in trade_dates:
            date_index = close.index.get_loc(dt)
            is_buy_date = dt in self.buy_dates
            is_sell_date = dt in self.sell_dates
            
            # Track date category
            if is_buy_date and is_sell_date:
                date_categories[dt] = 'BOTH'
            elif is_buy_date:
                date_categories[dt] = 'BUY'
            elif is_sell_date:
                date_categories[dt] = 'SELL'
            else:
                date_categories[dt] = 'HOLD'
            
            # === SELL LOGIC ===
            if is_sell_date and holdings:
                sells = set(self.get_sells(close, dt, holdings)) & holdings
                for sym in sells:
                    #weights.loc[dt, sym] = 0.0
                    weights_array[date_index, col_map[sym]] = 0
                    holdings.remove(sym)
                    self._orders.append(Order(
                        date=dt, symbol=sym, action='SELL',
                        price=close.loc[dt, sym], weight=0.0,
                        reason=f'sell_date_{self.sell_freq}'
                    ))
            
            # === BUY LOGIC ===
            if is_buy_date:
                needed = self.max_positions - len(holdings)
                if needed > 0:
                    buys = self.get_buys(close, dt, holdings, needed)
                    buys = set(buys) - holdings
                    for sym in list(buys)[:needed]:
                        #weights.loc[dt, sym] = weight_per_pos
                        weights_array[date_index, col_map[sym]] = weight_per_pos
                        holdings.add(sym)
                        self._orders.append(Order(
                            date=dt, symbol=sym, action='BUY',
                            price=close.loc[dt, sym], weight=weight_per_pos,
                            reason=f'buy_date_{self.buy_freq}'
                        ))
        

        weights = pd.DataFrame(weights_array, index=close.index, columns=close.columns)
        # Store date categories for debugging
        self._date_categories = date_categories
        
        return weights
    
    def get_orders(self) -> pd.DataFrame:
        """Get all orders as DataFrame."""
        if not self._orders:
            return pd.DataFrame()
        return pd.DataFrame([o.to_dict() for o in self._orders])
    
    def get_order_summary(self) -> Dict[str, Any]:
        """Get detailed order summary."""
        df = self.get_orders()
        if df.empty:
            return {
                'total': 0, 'buys': 0, 'sells': 0,
                'buy_dates': len(self.buy_dates),
                'sell_dates': len(self.sell_dates),
                'buy_freq': self.buy_freq,
                'sell_freq': self.sell_freq,
                'date_categories': {}
            }
        
        return {
            'total': len(df),
            'buys': len(df[df['action'] == 'BUY']),
            'sells': len(df[df['action'] == 'SELL']),
            'buy_dates': len(self.buy_dates),
            'sell_dates': len(self.sell_dates),
            'buy_freq': self.buy_freq,
            'sell_freq': self.sell_freq,
            'date_categories': {
                'both': sum(1 for v in self._date_categories.values() if v == 'BOTH'),
                'buy_only': sum(1 for v in self._date_categories.values() if v == 'BUY'),
                'sell_only': sum(1 for v in self._date_categories.values() if v == 'SELL'),
                'hold': sum(1 for v in self._date_categories.values() if v == 'HOLD')
            }
        }
    
    def get_trading_calendar(self) -> pd.DataFrame:
        """Get trading calendar with buy/sell dates marked."""
        if not hasattr(self, '_date_categories'):
            return pd.DataFrame()
        
        df = pd.DataFrame([
            {'date': dt, 'category': cat}
            for dt, cat in self._date_categories.items()
        ])
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        return df


class RSIStrategy(Strategy):
    """RSI strategy with configurable thresholds."""
    
    def __init__(self, rsi_window=200, buy_threshold=54, sell_threshold=51, **kwargs):
        # Keep accepting the old constructor name for direct callers.
        rsi_window = kwargs.pop("window", rsi_window)
        super().__init__(**kwargs)
        self.rsi_window = rsi_window
        self.buy_th, self.sell_th = buy_threshold, sell_threshold
        self.rsi = None
    
    def prepare(self, prices):
        delta = prices.diff()
        gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/self.rsi_window, adjust=False, min_periods=self.rsi_window).mean()
        avg_loss = loss.ewm(alpha=1/self.rsi_window, adjust=False, min_periods=self.rsi_window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        self.rsi = 100 - (100 / (1 + rs))
    
    def get_buys(self, prices, dt, holdings, needed):
        rsi = self.rsi.loc[dt]
        candidates = rsi[(rsi > self.buy_th) & (~rsi.index.isin(holdings))]
        return candidates.sort_values(ascending=False).head(needed).index.tolist()
    
    def get_sells(self, prices, dt, holdings):
        if not holdings:
            return []
        rsi = self.rsi.loc[dt]
        return [s for s in holdings if s in rsi.index and rsi[s] < self.sell_th]


class RSINewStrategy(Strategy):
    """RSI strategy with configurable thresholds."""
    
    def __init__(self, rsi_window=200, buy_rank=10, sell_rank=20, sell_threshold=51, **kwargs):
        # Keep accepting the old constructor name for direct callers.
        rsi_window = kwargs.pop("window", rsi_window)
        super().__init__(**kwargs)
        self.rsi_window = rsi_window
        self.buy_rank, self.sell_rank = buy_rank, sell_rank
        self.sell_th = sell_threshold

        self.rsi = None
    
    def prepare(self, prices):
        delta = prices.diff()
        gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/self.rsi_window, adjust=False, min_periods=self.rsi_window).mean()
        avg_loss = loss.ewm(alpha=1/self.rsi_window, adjust=False, min_periods=self.rsi_window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        self.rsi = 100 - (100 / (1 + rs))
        self.ranks = self.rsi.rank(axis=1, ascending=False, method='average')
    
    def get_buys(self, prices, dt, holdings, needed):
        if needed <= 0:
            return []
        rsi = self.rsi.loc[dt]

        top_candidates = rsi.nlargest(self.buy_rank)
        top_candidates = top_candidates[
            (top_candidates > self.sell_th) &
            (~top_candidates.index.isin(holdings))
        ]
        return top_candidates.head(needed).index.tolist()
    
    def get_sells(self, prices, dt, holdings):
        if not holdings:
            return []
        rsi, ranks = self.rsi.loc[dt], self.ranks.loc[dt]
        sells = []
        for s in holdings:
            if s not in rsi.index:
                continue
            if pd.notna(rsi[s]) and rsi[s] < self.sell_th:
                sells.append(s)
                continue
            if pd.isna(ranks[s]):
                sells.append(s)
                continue
            if ranks[s] > self.sell_rank:
                sells.append(s)
                continue
        return sells


class RSINewAndStrategy(Strategy):
    """RSI strategy with configurable thresholds."""
    
    def __init__(self, rsi_window=200, buy_rank=10, sell_rank=20, sell_threshold=51, **kwargs):
        # Keep accepting the old constructor name for direct callers.
        rsi_window = kwargs.pop("window", rsi_window)
        super().__init__(**kwargs)
        self.rsi_window = rsi_window
        self.buy_rank, self.sell_rank = buy_rank, sell_rank
        self.sell_th = sell_threshold

        self.rsi = None
    
    def prepare(self, prices):
        delta = prices.diff()
        gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/self.rsi_window, adjust=False, min_periods=self.rsi_window).mean()
        avg_loss = loss.ewm(alpha=1/self.rsi_window, adjust=False, min_periods=self.rsi_window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        self.rsi = 100 - (100 / (1 + rs))
        self.ranks = self.rsi.rank(axis=1, ascending=False, method='average')
    
    def get_buys(self, prices, dt, holdings, needed):
        if needed <= 0:
            return []
        rsi, ranks = self.rsi.loc[dt], self.ranks.loc[dt]
        candidates = ranks[
            rsi.notna() &
            (ranks <= self.buy_rank) & 
            (~ranks.index.isin(holdings)) &
            (rsi > 53)
        ]
        candidates = candidates.sort_values().head(needed).index.tolist()
        return candidates
    
    def get_sells(self, prices, dt, holdings):
        if not holdings:
            return []
        rsi, ranks = self.rsi.loc[dt], self.ranks.loc[dt]
        sells = []
        for s in holdings:
            if s not in rsi.index:
                continue
            if pd.notna(rsi[s]) and rsi[s] < self.sell_th and ranks[s] > self.sell_rank:
                sells.append(s)
                continue
            # if pd.isna(ranks[s]):
            #     sells.append(s)
            #     continue
            # if ranks[s] > self.sell_rank:
            #     sells.append(s)
            #     continue
        return sells


class MomentumStrategy(Strategy):
    """Relative momentum strategy with ranking."""
    
    def __init__(
        self,
        lookback=256,
        skip_months=1,
        buy_rank=10,
        sell_rank=15,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lookback, self.buy_rank, self.sell_rank = lookback, buy_rank, sell_rank
        self.ranks = None
        self.skip_months = skip_months
        self.skip_days = int(skip_months * 21)
    
    def prepare(self, prices):
        ret = prices.shift(self.skip_days).pct_change(self.lookback)
        self.ranks = ret.rank(axis=1, ascending=False)
    
    def get_buys(self, prices, dt, holdings, needed):
        ranks = self.ranks.loc[dt]
        candidates = ranks[(ranks <= self.buy_rank) & (~ranks.index.isin(holdings))]
        return candidates.nsmallest(needed).index.tolist()

    
    def get_sells(self, prices, dt, holdings):
        if not holdings:
            return []
        ranks = self.ranks.loc[dt]
        return [
            stock for stock in holdings
            if ranks.get(stock, float("inf")) > self.sell_rank
        ] 
        #s for s in holdings if s in ranks.index and ranks[s] > self.sell_rank]


class DualMomentumStrategy(Strategy):
    """Relative momentum strategy with ranking."""
    
    def __init__(self, lookback=256, skip=21, buy_rank=10, sell_rank=15, **kwargs):
        super().__init__(**kwargs)
        self.lookback, self.buy_rank, self.sell_rank = lookback, buy_rank, sell_rank
        self.ranks = None
        self.roc = None
        self.skip = skip
        self.ma200 = None
    
    def prepare(self, prices):
        self.roc = prices.shift(self.skip).pct_change(self.lookback)
        self.ranks = self.roc.rank(axis=1, ascending=False)
        self.ma200 = prices.rolling(200).mean()
    
    def get_buys(self, prices, dt, holdings, needed):
        if needed <= 0:
            return []

        roc_now = self.roc.loc[dt]
        positive_roc = roc_now[roc_now > 0].dropna()
        if positive_roc.empty:
            return []

        ranks = self.ranks.loc[dt].reindex(positive_roc.index)
        candidates = ranks[
            (~ranks.index.isin(holdings)) &
            (ranks <= self.buy_rank)
        ]
        if candidates.empty:
            return []

        price_now = prices.loc[dt]
        ma200_now = self.ma200.loc[dt]
        candidates = candidates[
            price_now[candidates.index] > (ma200_now[candidates.index] * 1.03)
        ]

        return candidates.nsmallest(needed).index.tolist()
   
    def get_sells(self, prices, dt, holdings):
        if not holdings:
            return []
        ranks = self.ranks.loc[dt]
        sells = []
        for s in holdings:
            if s not in ranks.index:
                continue

            price_now = prices.loc[dt, s]
            ma200_now = self.ma200.loc[dt, s]
            ma200_stop = (
                pd.notna(price_now) 
                and pd.notna(ma200_now)
                and price_now < ma200_now *.98
            )

            rank_sell = ranks[s] > self.sell_rank
            roc_series = self.roc[s]
            roc_now = roc_series.loc[dt]
            roc_20 = roc_series.shift(30).loc[dt]
            roc_40 = roc_series.shift(60).loc[dt]
            trend_sell = (
                pd.notna(roc_now)
                and pd.notna(roc_20)
                and pd.notna(roc_40)
                and (roc_now < roc_20)
                and (roc_20 < roc_40)
            )

            if ma200_stop or rank_sell :
                sells.append(s)

        return sells


class StrategyFactory:
    """Strategy factory."""
    _registry = {
        'rsi': RSIStrategy,
        'momentum': MomentumStrategy,
        'momentum2': MomentumStrategy,
        'dualmomentum': DualMomentumStrategy,
        'dual_momentum': DualMomentumStrategy,
        'rsi_new': RSINewStrategy,
        'rsi_new_and': RSINewAndStrategy,
    }
    
    @classmethod
    def create(cls, name: str, **kwargs):
        key = name.lower()
        if key not in cls._registry:
            raise KeyError(f"Unknown strategy. Available: {list(cls._registry.keys())}")
        strategy_class = cls._registry[key]
        return strategy_class(**kwargs)

    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> Strategy :
        name = config.get('name')
        if not name:
            raise ValueError("Config must have 'name' key.")
        
        params = config.get('params', {})
        return cls.create(name, params)

    @classmethod
    def list_strategies(cls) -> List[str]:
        return list(cls._registry.keys())



