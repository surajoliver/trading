from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class StockDecision:
    sell: list
    buy: list
    hold: list
    available_slots: int
    ranks: pd.Series | None = field(default=None, repr=False)
    prices: pd.Series | None = field(default=None, repr=False)
    ma: pd.Series | None = field(default=None, repr=False)
    ma50: pd.Series | None = field(default=None, repr=False)
    roc: pd.Series | None = field(default=None, repr=False)
    roc_current: pd.Series | None = field(default=None, repr=False)
    ranks_current: pd.Series | None = field(default=None, repr=False)
    roc_history: dict | None = field(default=None, repr=False)
    rank_history: dict | None = field(default=None, repr=False)
    macd: pd.Series | None = field(default=None, repr=False)
    macd_histogram: pd.Series | None = field(default=None, repr=False)
    accel_6m_4w: pd.Series | None = field(default=None, repr=False)
    sell_rank: int | None = field(default=None, repr=False)
    entry_prices: dict | None = field(default=None, repr=False)
    entry_dates: dict | None = field(default=None, repr=False)
    stop_loss_prices: dict | None = field(default=None, repr=False)
    mae: dict | None = field(default=None, repr=False)
    mfe: dict | None = field(default=None, repr=False)
    highest_prices: dict | None = field(default=None, repr=False)
    trailing_stop_prices: dict | None = field(default=None, repr=False)
    sell_reasons: dict | None = field(default=None, repr=False)

    def _status_for_symbol(self, symbol):
        if symbol in self.buy:
            return "Buy"
        if symbol in self.sell:
            return "Sell"
        if symbol in self.hold:
            return "Hold"

        if self.sell_rank is not None:
            if self.ranks is not None and symbol in self.ranks.index and self.ranks[symbol] > self.sell_rank:
                return "Sell"
            if self.ranks_current is not None and symbol in self.ranks_current.index and self.ranks_current[symbol] > self.sell_rank:
                return "Sell"

        return "None"

    def _build_rows(self):
        rows = []
        all_symbols = list(dict.fromkeys(self.buy + self.hold + self.sell))

        for symbol in all_symbols:
            rank = self.ranks.get(symbol) if self.ranks is not None and symbol in self.ranks.index else None
            close_price = self.prices.get(symbol) if self.prices is not None and symbol in self.prices.index else None
            ma_price = self.ma.get(symbol) if self.ma is not None and symbol in self.ma.index else None
            ma50_price = self.ma50.get(symbol) if self.ma50 is not None and symbol in self.ma50.index else None
            status = self._status_for_symbol(symbol)

            last_price = round(close_price, 2) if close_price is not None else None
            ma200 = round(ma_price, 2) if ma_price is not None else None
            ma50 = round(ma50_price, 2) if ma50_price is not None else None
            entry_price = self.entry_prices.get(symbol) if self.entry_prices else None
            entry_date = self.entry_dates.get(symbol) if self.entry_dates else None
            stop_loss_price = self.stop_loss_prices.get(symbol) if self.stop_loss_prices else None
            mae = self.mae.get(symbol) if self.mae else None
            mfe = self.mfe.get(symbol) if self.mfe else None
            highest_price = self.highest_prices.get(symbol) if self.highest_prices else None
            trailing_stop_price = self.trailing_stop_prices.get(symbol) if self.trailing_stop_prices else None
            roc_value = self.roc.get(symbol) if self.roc is not None and symbol in self.roc.index else None
            roc_current_value = self.roc_current.get(symbol) if self.roc_current is not None and symbol in self.roc_current.index else None
            rank_current_value = self.ranks_current.get(symbol) if self.ranks_current is not None and symbol in self.ranks_current.index else None
            macd_value = self.macd.get(symbol) if self.macd is not None and symbol in self.macd.index else None
            macd_hist_value = self.macd_histogram.get(symbol) if self.macd_histogram is not None and symbol in self.macd_histogram.index else None
            accel_value = self.accel_6m_4w.get(symbol) if self.accel_6m_4w is not None and symbol in self.accel_6m_4w.index else None

            if roc_value is not None:
                roc_value = round(roc_value * 100, 2)
            if roc_current_value is not None:
                roc_current_value = round(roc_current_value * 100, 2)
            if macd_value is not None:
                macd_value = round(macd_value, 2)
            if macd_hist_value is not None:
                macd_hist_value = round(macd_hist_value, 2)
            if accel_value is not None:
                accel_value = round(accel_value, 2)

            roc_history = self.roc_history.get(symbol) if self.roc_history is not None and symbol in self.roc_history else None
            rank_history = self.rank_history.get(symbol) if self.rank_history is not None and symbol in self.rank_history else None

            if close_price is not None and ma_price is not None and ma_price != 0:
                extension_over_ma200 = round(((close_price - ma_price) / ma_price) * 100, 2)
            else:
                extension_over_ma200 = None

            if close_price is not None and ma50_price is not None and ma50_price != 0:
                extension_over_ma50 = round(((close_price - ma50_price) / ma50_price) * 100, 2)
            else:
                extension_over_ma50 = None

            if status == "Buy":
                qty_to_buy = round(70000 / close_price) if close_price not in (None, 0) else 0
                if rank is not None and ma200 is not None:
                    remarks = f"Rank {rank}; ROC {roc_value}% ; MA200 {ma200}; +{extension_over_ma200}% over MA200; MA50 {ma50}; +{extension_over_ma50}% over MA50; MACD {macd_value}; accel {accel_value}pp; price supports fresh entry."
                else:
                    remarks = "Rank/MA200 not available; eligible for entry."
            elif status == "Sell":
                qty_to_buy = 0
                sell_reason = self.sell_reasons.get(symbol, "Momentum weak") if self.sell_reasons else "Momentum weak"
                if rank is not None and ma200 is not None:
                    remarks = f"Rank {rank}; ROC {roc_value}% ; MA200 {ma200}; +{extension_over_ma200}% over MA200; MA50 {ma50}; +{extension_over_ma50}% over MA50; MACD {macd_value}; accel {accel_value}pp; {sell_reason}, exit position."
                else:
                    remarks = f"{sell_reason}, exit position."
            elif status == "Hold":
                qty_to_buy = 0
                if rank is not None and ma200 is not None:
                    remarks = f"Rank {rank}; ROC {roc_value}% ; MA200 {ma200}; +{extension_over_ma200}% over MA200; MA50 {ma50}; +{extension_over_ma50}% over MA50; MACD {macd_value}; accel {accel_value}pp; continue holding as trend remains intact."
                else:
                    remarks = "Continue holding as trend remains intact."
            else:
                qty_to_buy = 0
                remarks = "No action required."

            rows.append({
                "stock": symbol,
                "status": status,
                "last_price": last_price,
                "entry_price": entry_price,
                "entry_date": entry_date,
                "stop_loss_price": stop_loss_price,
                "mae": mae,
                "mfe": mfe,
                "highest_price": highest_price,
                "trailing_stop_price": trailing_stop_price,
                "qty_to_buy": qty_to_buy,
                "rank": rank,
                "roc": roc_value,
                "rank_current_month": rank_current_value,
                "roc_current_month": roc_current_value,
                "roc_history": roc_history,
                "rank_history": rank_history,
                "ma200": ma200,
                "extension": extension_over_ma200,
                "ma50": ma50,
                "extension2": extension_over_ma50,
                "macd": macd_value,
                "macd_histogram": macd_hist_value,
                "accel_6m_4w": accel_value,
                "remarks": remarks,
            })

        return rows

    def to_dataframe(self):
        rows = self._build_rows()
        if not rows:
            return pd.DataFrame(columns=["stock", "status", "last_price", "entry_price", "entry_date", "stop_loss_price", "mae", "mfe", "highest_price", "trailing_stop_price", "qty_to_buy", "rank", "roc", "rank_current_month", "roc_current_month", "roc_history", "rank_history", "ma200", "extension", "ma50", "extension2", "macd", "macd_histogram", "accel_6m_4w", "remarks"])

        df = pd.DataFrame(rows)
        status_order = {"Buy": 0, "Hold": 1, "Sell": 2}
        df["status_order"] = df["status"].map(status_order).fillna(99)
        df = df.sort_values(["status_order", "rank"], na_position="last").reset_index(drop=True)
        return df.drop(columns=["status_order"])

    def __str__(self):
        df = self.to_dataframe()
        if df.empty:
            return "Empty StockDecision"
        return df.to_string(index=False)

    def __repr__(self):
        return self.__str__()


class MomentumLiveSelector:
    def __init__(
        self,
        lookback=256,
        skip=21,
        sell_rank=15,
        max_positions=10,
        ma_window=200,
        ma_buffer=0.0,
    ):
        self.lookback = lookback
        self.skip = skip
        self.sell_rank = sell_rank
        self.max_positions = max_positions
        self.ma_window = ma_window
        self.ma_buffer = ma_buffer

    def prepare(self, close):
        self.roc = close.shift(self.skip).pct_change(self.lookback)
        self.ranks = self.roc.rank(axis=1, ascending=False)
        self.roc_current = close.pct_change(self.lookback)
        self.ranks_current = self.roc_current.rank(axis=1, ascending=False)
        self.ma = close.rolling(self.ma_window).mean()
        self.ma50 = close.rolling(50).mean()

        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        self.macd = ema50 - ema200
        self.macd_signal = self.macd.ewm(span=9, adjust=False).mean()
        self.macd_histogram = self.macd - self.macd_signal

        macd_scale = self.macd.abs().rolling(20, min_periods=1).max().replace(0, np.nan)
        hist_scale = self.macd_histogram.abs().rolling(20, min_periods=1).max().replace(0, np.nan)

        self.macd_norm = (self.macd / macd_scale).clip(lower=-1, upper=1)
        self.macd_histogram_norm = (self.macd_histogram / hist_scale).clip(lower=-1, upper=1)

        roc_6m_pct = close.pct_change(126) * 100
        roc_6m_4w_pct = roc_6m_pct.shift(20)
        self.accel_6m_4w = roc_6m_pct - roc_6m_4w_pct

    def select(self, close, holdings, dt=None, entry_prices=None, entry_dates=None, stop_loss_pct=0.20):
        dt = dt or close.index[-1]

        ranks = self.ranks.loc[dt].dropna()
        ranks_current = self.ranks_current.loc[dt].dropna() if self.ranks_current is not None else ranks
        price = close.loc[dt]
        ma = self.ma.loc[dt]
        ma50 = self.ma50.loc[dt]
        roc = self.roc.loc[dt]
        roc_current_series = self.roc_current.loc[dt] if self.roc_current is not None else roc
        macd = self.macd_norm.loc[dt]
        macd_histogram = self.macd_histogram_norm.loc[dt]
        accel_6m_4w = self.accel_6m_4w.loc[dt]

        def _format_history(current, prev_1m, prev_2m, decimals=2, as_int=False):
            def fmt(value):
                if value is None or pd.isna(value):
                    return "n/a"
                if as_int:
                    return str(int(round(float(value))))
                return str(round(float(value), decimals))

            return " <- ".join([fmt(current), fmt(prev_1m), fmt(prev_2m)])

        roc_history = {}
        rank_history = {}
        for symbol in ranks.index:
            roc_series = self.roc[symbol]
            rank_series = self.ranks[symbol]

            roc_latest = roc_series.iloc[-1] if len(roc_series) else None
            roc_prev_1m = roc_series.shift(30).iloc[-1] if len(roc_series) else None
            roc_prev_2m = roc_series.shift(60).iloc[-1] if len(roc_series) else None
            roc_history[symbol] = _format_history(
                roc_latest * 100 if roc_latest is not None else None,
                roc_prev_1m * 100 if roc_prev_1m is not None else None,
                roc_prev_2m * 100 if roc_prev_2m is not None else None,
                decimals=2,
            )

            rank_current = rank_series.iloc[-1] if len(rank_series) else None
            rank_prev_1m = rank_series.shift(30).iloc[-1] if len(rank_series) else None
            rank_prev_2m = rank_series.shift(60).iloc[-1] if len(rank_series) else None
            rank_history[symbol] = _format_history(rank_current, rank_prev_1m, rank_prev_2m, decimals=0, as_int=True)

        holdings = list(holdings)
        entry_prices = entry_prices or {}
        entry_dates = entry_dates or {}
        stop_loss_prices = {
            stock: round(float(entry_prices[stock]) * (1 - stop_loss_pct), 2)
            for stock in holdings
            if stock in entry_prices and pd.notna(entry_prices[stock])
        }
        mae = {}
        mfe = {}
        highest_prices = {}
        trailing_stop_prices = {}
        for stock in holdings:
            if stock not in entry_prices or pd.isna(entry_prices[stock]) or stock not in close.columns:
                continue

            entry_date = entry_dates.get(stock)
            if entry_date is None or pd.isna(entry_date):
                continue

            try:
                entry_date = pd.Timestamp(entry_date)
                entry_price = float(entry_prices[stock])
            except (TypeError, ValueError):
                continue

            prices_since_entry = close.loc[entry_date:dt, stock].dropna()
            if prices_since_entry.empty or entry_price <= 0:
                continue

            lowest_price = float(prices_since_entry.min())
            highest_price = float(prices_since_entry.max())
            mae[stock] = round(max(0.0, (entry_price - lowest_price) / entry_price * 100), 2)
            mfe[stock] = round(max(0.0, (highest_price - entry_price) / entry_price * 100), 2)
            highest_prices[stock] = round(highest_price, 2)
            trailing_stop_prices[stock] = round(highest_price * (1 - stop_loss_pct), 2)

        # Existing positions: sell first
        momentum_sell = [
            stock for stock in holdings
            if stock in ranks.index
            and (
                ranks[stock] > self.sell_rank
                or (stock in ranks_current.index and ranks_current[stock] > self.sell_rank)
            )
        ]
        stop_loss_sell = [
            stock for stock in holdings
            if stock in stop_loss_prices
            and stock in price.index
            and pd.notna(price[stock])
            and price[stock] <= stop_loss_prices[stock]
        ]
        trailing_stop_sell = [
            stock for stock in holdings
            if stock in trailing_stop_prices
            and stock in price.index
            and pd.notna(price[stock])
            and price[stock] <= trailing_stop_prices[stock]
        ]
        sell = list(dict.fromkeys(momentum_sell + stop_loss_sell + trailing_stop_sell))
        sell_reasons = {
            stock: "; ".join(reason for reason, triggered in (
                ("stop loss hit", stock in stop_loss_sell),
                ("trailing stop hit", stock in trailing_stop_sell),
                ("rank threshold breached", stock in momentum_sell),
            ) if triggered)
            for stock in sell
        }

        hold = [s for s in holdings if s not in sell]

        slots = self.max_positions - len(hold)

        if slots <= 0:
            return StockDecision(sell, [], hold, 0, ranks=ranks, prices=price, ma=ma, ma50=ma50, roc=roc, roc_current=roc_current_series, ranks_current=ranks_current, roc_history=roc_history, rank_history=rank_history, macd=macd, macd_histogram=macd_histogram, accel_6m_4w=accel_6m_4w, sell_rank=self.sell_rank, entry_prices=entry_prices, entry_dates=entry_dates, stop_loss_prices=stop_loss_prices, mae=mae, mfe=mfe, highest_prices=highest_prices, trailing_stop_prices=trailing_stop_prices, sell_reasons=sell_reasons)

        # New entries
        candidates = ranks[
            ~ranks.index.isin(hold)
            & ~ranks.index.isin(sell)
        ]

        eligible = candidates[
            price.reindex(candidates.index)
            .gt(
                ma.reindex(candidates.index) *
                (1 + self.ma_buffer)
            )
        ]

        buy = eligible.nsmallest(slots).index.tolist()

        return StockDecision(
            sell=sell,
            buy=buy,
            hold=hold,
            available_slots=slots - len(buy),
            ranks=ranks,
            prices=price,
            ma=ma,
            ma50=ma50,
            roc=roc,
            roc_current=roc_current_series,
            ranks_current=ranks_current,
            roc_history=roc_history,
            rank_history=rank_history,
            macd=macd,
            macd_histogram=macd_histogram,
            accel_6m_4w=accel_6m_4w,
            sell_rank=self.sell_rank,
            entry_prices=entry_prices,
            entry_dates=entry_dates,
            stop_loss_prices=stop_loss_prices,
            mae=mae,
            mfe=mfe,
            highest_prices=highest_prices,
            trailing_stop_prices=trailing_stop_prices,
            sell_reasons=sell_reasons,
        )