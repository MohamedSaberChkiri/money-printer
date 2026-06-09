import logging
import time
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import linregress
import os

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class HalalMaxGrowthEngineV3:
    def __init__(self, api_key, secret_key, paper=True):
        self.trading = TradingClient(api_key, secret_key, paper=paper)
        self.data = StockHistoricalDataClient(api_key, secret_key)
        
        self.universe = ["AAPL", "MSFT", "NVDA", "AMD", "META", "AMZN", "GOOGL", "TSM", "AVGO", "LLY"]
        self.benchmark = "SPY"
        self.long_window = 126
        self.mid_window = 63
        self.short_window = 21
        self.max_position_weight = 0.95

    def momentum(self, prices):
        logp = np.log(prices)
        x = np.arange(len(logp))
        slope, _, r, _, _ = linregress(x, logp)
        return slope * 252 * (r ** 2)

    def signal(self, series):
        if len(series.dropna()) < self.long_window: return None
        p = series.dropna().values
        long = self.momentum(p[-self.long_window:])
        mid = self.momentum(p[-self.mid_window:])
        short = self.momentum(p[-self.short_window:])
        accel = (short - mid) + 0.5 * (mid - long)
        alignment = np.sign(long) == np.sign(mid) == np.sign(short)
        return (long + accel) * (1.4 if alignment else 0.6)

    def regime(self, df):
        px = df[self.benchmark]
        sma = px.rolling(200).mean().iloc[-1]
        trend = (px.iloc[-1] / sma) - 1
        vol = max(px.pct_change().rolling(20).std().iloc[-1], 1e-6)
        return trend / vol

    def exposure_mode(self, regime):
        if regime > 1.0: return 1.0
        elif regime > 0.3: return 0.8
        elif regime > 0: return 0.6
        elif regime > -0.3: return 0.4
        else: return 0.2

    def fetch(self):
        end = datetime.now()
        start = end - timedelta(days=365)
        req = StockBarsRequest(
            symbol_or_symbols=self.universe + [self.benchmark],
            timeframe=TimeFrame.Day,
            start=start, end=end
        )
        return self.data.get_stock_bars(req).df["close"].unstack(level=0).ffill()

    def run(self):
        clock = self.trading.get_clock()
        if not clock.is_open:
            logging.info("Market closed. Sleeping.")
            return

        df = self.fetch()
        regime = self.regime(df)
        exposure = self.exposure_mode(regime)
        
        signals = {sym: self.signal(df[sym]) for sym in self.universe if self.signal(df[sym]) is not None}
        ranked = sorted(signals, key=signals.get, reverse=True)
        top_k = 1 if regime > 1.0 else (2 if regime > 0.3 else 3)
        top_symbols = ranked[:top_k]

        # Calculate target weights
        raw_vals = {t: signals[t] for t in top_symbols}
        total = sum(abs(v) for v in raw_vals.values())
        target_weights = {k: min((abs(v) / total), self.max_position_weight) for k, v in raw_vals.items()}

        # Get cash and apply safety buffer
        cash = float(self.trading.get_account().cash)
        capital = (cash * exposure) * 0.98 

        logging.info(f"REGIME={regime:.2f} EXPOSURE={exposure:.2f} TARGETS={top_symbols}")

        # 1. Close positions not in targets
        current_positions = self.trading.get_all_positions()
        for pos in current_positions:
            if pos.symbol not in top_symbols:
                logging.info(f"Closing {pos.symbol}")
                self.trading.close_position(pos.symbol)

        # 2. Buy/Rebalance
        for sym in top_symbols:
            price = df[sym].iloc[-1]
            target_qty = (capital * target_weights[sym]) / price
            
            # Simple approach: Buy/Fill to target
            try:
                self.trading.submit_order(
                    MarketOrderRequest(
                        symbol=sym,
                        qty=round(target_qty, 4),
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    )
                )
                logging.info(f"Ordered {sym} qty={target_qty:.4f}")
            except Exception as e:
                logging.error(f"Order failed {sym}: {e}")

if __name__ == "__main__":

    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    bot = HalalMaxGrowthEngineV3(api_key, secret_key, paper=True)
    bot.run()