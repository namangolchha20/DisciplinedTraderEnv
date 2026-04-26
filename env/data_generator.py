import numpy as np
from typing import List
from .models import OHLCV

def generate_synthetic_ohlcv(start_price=100.0, num_bars=5000, volatility=0.01, trend=0.0001):
    prices = [start_price]
    for _ in range(num_bars-1):
        change = np.random.normal(trend, volatility)
        prices.append(prices[-1] * (1 + change))
    ohlcv_list = []
    for i in range(num_bars):
        open_p = prices[i]
        high = open_p * (1 + abs(np.random.normal(0, volatility/2)))
        low = open_p * (1 - abs(np.random.normal(0, volatility/2)))
        close = (high + low) / 2 + np.random.normal(0, volatility*open_p/10)
        volume = np.random.uniform(100, 1000)
        ohlcv_list.append(OHLCV(open=open_p, high=high, low=low, close=close, volume=volume))
    return ohlcv_list

def resample_ohlcv(ohlcv_list: List[OHLCV], factor: int) -> List[OHLCV]:
    """Aggregate every `factor` bars into one."""
    resampled = []
    for i in range(0, len(ohlcv_list), factor):
        chunk = ohlcv_list[i:i+factor]
        if not chunk:
            continue
        o = chunk[0].open
        c = chunk[-1].close
        h = max(x.high for x in chunk)
        l = min(x.low for x in chunk)
        v = sum(x.volume for x in chunk)
        resampled.append(OHLCV(open=o, high=h, low=l, close=c, volume=v))
    return resampled

# Resampling factors (for 1m base data)
RESAMPLE_FACTORS = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "1d": 390   # 6.5 hours * 60 minutes
}