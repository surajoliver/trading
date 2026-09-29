"""Trading strategy implementations."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Set
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
        self.entry_prices = {}
        self.peak_prices = {}
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
                for sym in holdings:
                    price = close.at[dt, sym]
                    if pd.notna(price):
                        self.peak_prices[sym] = max(self.peak_prices.get(sym, price), price)
                sells = set(self.get_sells(close, dt, holdings)) & holdings
                for sym in sells:
                    #weights.loc[dt, sym] = 0.0
                    weights_array[date_index, col_map[sym]] = 0
                    holdings.remove(sym)
                    self.entry_prices.pop(sym, None)
                    self.peak_prices.pop(sym, None)
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
                        self.entry_prices[sym] = close.at[dt, sym]
                        self.peak_prices[sym] = close.at[dt, sym]
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


class MomentumStrategy(Strategy):
    """Relative momentum strategy with ranking."""
    
    def __init__(
        self,
        lookback=256,
        skip_months=1,
        buy_rank=None,
        sell_rank=15,
        ma_window=None,
        initial_stop_loss_pct=None,
        peak_stop_loss_pct=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lookback, self.buy_rank, self.sell_rank = lookback, buy_rank, sell_rank
        self.ranks = None
        self.skip_months = skip_months
        self.skip_days = int(skip_months * 21)
        self.ma_window = ma_window
        self.ma = None
        self.initial_stop_loss_pct = initial_stop_loss_pct
        self.peak_stop_loss_pct = peak_stop_loss_pct
        print('stop loss: ', self.initial_stop_loss_pct)
    
    def prepare(self, prices):
        ret = prices.shift(self.skip_days).pct_change(self.lookback)
        self.ranks = ret.rank(axis=1, ascending=False)
        self.ma = prices.rolling(self.ma_window).mean() if self.ma_window else None
    
    def get_buys(self, prices, dt, holdings, needed):
        ranks = self.ranks.loc[dt]
        candidates = ranks[~ranks.index.isin(holdings)]
        if self.buy_rank is not None:
            candidates = candidates[candidates <= self.buy_rank]
        if self.ma is not None:
            above_ma = prices.loc[dt].gt(self.ma.loc[dt])
            candidates = candidates[above_ma.reindex(candidates.index, fill_value=False)]
        return candidates.nsmallest(needed).index.tolist()

    
    def get_sells(self, prices, dt, holdings):
        if not holdings:
            return []
        ranks = self.ranks.loc[dt]
        sells = [
            stock for stock in holdings
            if ranks.get(stock, float("inf")) > self.sell_rank
        ]

        current_prices = prices.loc[dt]
        if self.initial_stop_loss_pct is not None:
            sells.extend(
                stock for stock in holdings
                if stock in self.entry_prices
                and pd.notna(current_prices.get(stock))
                and current_prices[stock] <= self.entry_prices[stock] * (1 - self.initial_stop_loss_pct)
            )
        if self.peak_stop_loss_pct is not None:
            sells.extend(
                stock for stock in holdings
                if stock in self.peak_prices
                and pd.notna(current_prices.get(stock))
                and current_prices[stock] <= self.peak_prices[stock] * (1 - self.peak_stop_loss_pct)
            )
        return list(dict.fromkeys(sells))


class MomentumScoreStrategy(MomentumStrategy):
    """Momentum ranked by the average percentile of multiple ROC horizons."""

    def __init__(self, roc_periods=(60, 120, 250), skip_months=1, **kwargs):
        super().__init__(skip_months=skip_months, **kwargs)
        self.roc_periods = tuple(roc_periods)
        self.scores = None

    def prepare(self, prices):
        skip = int(self.skip_months * 21)
        percentiles = [
            prices.shift(skip).pct_change(period).rank(axis=1, pct=True)
            for period in self.roc_periods
        ]
        self.scores = sum(percentiles) / len(percentiles)
        self.ranks = self.scores.rank(axis=1, ascending=False)


class MomentumStopLossStrategy(MomentumStrategy):
    """Momentum strategy with initial and peak-based stop losses."""

    def __init__(
        self,
        initial_stop_loss_pct=0.20,
        peak_stop_loss_pct=0.20,
        **kwargs,
    ):
        super().__init__(
            initial_stop_loss_pct=initial_stop_loss_pct,
            peak_stop_loss_pct=peak_stop_loss_pct,
            **kwargs,
        )

class ProportionalROCIvStrategy(MomentumStrategy):
    """
    Momentum strategy ranked by Proportional ROC (Rate of Change) and Inverse Volatility.
    
    Metric = ROC * Inverse Volatility = ROC / Volatility
    This acts as a risk-adjusted momentum metric (similar to a Sharpe-like ranking factor).
    """

    def __init__(
        self,
        lookback: int = 252,
        vol_window: int = 90,
        skip_months: float = 1.0,
        **kwargs,
    ):
        """
        Parameters:
        -----------
        lookback : int
            Number of periods/days to calculate the Rate of Change (ROC).
        vol_window : int
            Rolling window (in days) to calculate volatility (standard deviation of daily returns).
        skip_months : float
            Skip period (in months) to avoid short-term reversal effects.
        """
        super().__init__(lookback=lookback, skip_months=skip_months, **kwargs)
        self.vol_window = vol_window
        self.scores = None

    def prepare(self, prices: pd.DataFrame):
        skip = self.skip_days

        # 1. Calculate Rate of Change (ROC)
        roc = prices.shift(skip).pct_change(self.lookback)

        # 2. Calculate Volatility (std of daily pct returns) over the specified window
        daily_returns = prices.pct_change()
        volatility = daily_returns.rolling(window=self.vol_window).std()

        # Shift volatility to align with the skipped window
        volatility = volatility.shift(skip)

        # 3. Calculate Inverse Volatility (1 / Volatility)
        # Avoid division by zero by replacing zero/extremely tiny values with NaN
        inverse_volatility = 1.0 / volatility.replace(0, np.nan)

        # 4. Proportional ROC * Inverse Volatility Factor (Risk-Adjusted Momentum)
        self.scores = roc * inverse_volatility

        # 5. Rank stocks across the universe for each date (Descending: rank 1 = best stock)
        self.ranks = self.scores.rank(axis=1, ascending=False)

        # Optional MA Filter initialization from base class
        if self.ma_window:
            self.ma = prices.rolling(self.ma_window).mean()

class StrategyFactory:
    """Strategy factory."""
    _registry = {
        # 'rsi': RSIStrategy,
        # 'momentum_monthly': MomentumStrategy,
        'momentum_quaterly': MomentumStrategy,
        # 'momentum_yearly': MomentumStrategy,
        # 'momentum_yearly_30': MomentumStrategy,        
        # # 'momentum_score': MomentumScoreStrategy,
        # 'momentum_stop_loss': MomentumStopLossStrategy,
        # 'momentum_stop_loss_05': MomentumStopLossStrategy,
        # 'momentum_stop_loss_10': MomentumStopLossStrategy,
        'momentum_stop_loss_15': MomentumStopLossStrategy,
        'momentum_roc_inv_vol': ProportionalROCIvStrategy,
        'momentum_custom': ProportionalROCIvStrategy,
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



