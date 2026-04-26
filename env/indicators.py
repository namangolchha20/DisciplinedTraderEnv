import numpy as np
from .patterns_v2 import detect_chart_pattern_v2 as detect_chart_pattern
from typing import List
from .models import OHLCV

def compute_rsi(prices: list, period=14):
    if len(prices) < period+1:
        return 50.0
    deltas = np.diff(prices[-period-1:])
    gains = deltas[deltas > 0].sum() / period
    losses = -deltas[deltas < 0].sum() / period
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100 - (100 / (1 + rs))

def compute_sma(prices: list, period=20):
    if len(prices) < period:
        return prices[-1]
    return np.mean(prices[-period:])

def compute_bollinger_bands(prices: list, period=20, std_dev=2):
    sma = compute_sma(prices, period)
    if len(prices) < period:
        return sma, sma, sma
    std = np.std(prices[-period:])
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower

def compute_super_trend(highs: list, lows: list, closes: list, period=10, multiplier=3):
    """Simplified Super Trend indicator."""
    if len(closes) < period:
        return closes[-1], 0
    atr = np.mean([highs[i] - lows[i] for i in range(-period, 0)])
    basic_upper = (highs[-1] + lows[-1]) / 2 + multiplier * atr
    basic_lower = (highs[-1] + lows[-1]) / 2 - multiplier * atr
    # Simple directional logic based on previous close
    if closes[-1] > basic_upper:
        return basic_lower, 1   # uptrend (buy)
    elif closes[-1] < basic_lower:
        return basic_upper, -1  # downtrend (sell)
    else:
        return basic_upper if closes[-1] > closes[-2] else basic_lower, (1 if closes[-1] > closes[-2] else -1)

def compute_volume_index(volumes: list, period=20):
    """Volume ratio: current volume / average volume over period."""
    if len(volumes) < period:
        return 1.0
    avg_vol = np.mean(volumes[-period:])
    if avg_vol == 0:
        return 1.0
    return volumes[-1] / avg_vol

# Keep your support/resistance and candlestick detection (unchanged)
def detect_support_resistance(highs, lows, lookback=20):
    support = min(lows[-lookback:])
    resistance = max(highs[-lookback:])
    return support, resistance

from typing import List
from .models import OHLCV

def detect_candlestick_pattern(bars: List[OHLCV], idx: int) -> str:
    """Detect candlestick patterns using the current and previous bars."""
    if idx < 0 or idx >= len(bars):
        return "none"
    bar = bars[idx]
    o = bar.open
    h = bar.high
    l = bar.low
    c = bar.close
    body = abs(c - o)
    total_range = h - l
    if total_range == 0:
        return "none"

    # --------------------------------------------------------------
    # Single bar patterns
    # --------------------------------------------------------------
    # Doji variants
    is_doji = body / total_range < 0.1
    if is_doji:
        # Dragonfly: lower wick long, upper wick short
        if (o - l) > 2 * (h - o) and (h - o) < (o - l) * 0.3 and c >= o:
            return "dragonfly_doji"
        # Gravestone: upper wick long, lower wick short
        if (h - o) > 2 * (o - l) and (o - l) < (h - o) * 0.3 and c <= o:
            return "gravestone_doji"
        return "doji"  # plain doji

    # Hammer / Hanging Man / Inverted Hammer / Shooting Star
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    # Hammer (bullish) / Hanging Man (bearish): small body, long lower wick, little upper wick
    if lower_wick > 2 * body and upper_wick < body:
        if c > o:
            return "hammer"   # bullish
        else:
            return "hanging_man"  # bearish
    # Inverted Hammer (bullish) / Shooting Star (bearish): small body, long upper wick, little lower wick
    if upper_wick > 2 * body and lower_wick < body:
        if c > o:
            return "inverted_hammer"  # bullish reversal potential
        else:
            return "shooting_star"    # bearish reversal

    # Marubozu: no or very small wicks
    if upper_wick < total_range * 0.05 and lower_wick < total_range * 0.05:
        if c > o:
            return "bullish_marubozu"
        else:
            return "bearish_marubozu"

    # --------------------------------------------------------------
    # Two-bar patterns (need previous bar)
    # --------------------------------------------------------------
    if idx < 1:
        return "none"
    prev = bars[idx-1]
    o1, c1 = prev.open, prev.close
    # Engulfing
    if c > o and c1 < o1 and c > o1 and o < c1:
        return "bullish_engulfing"
    if c < o and c1 > o1 and c < o1 and o > c1:
        return "bearish_engulfing"

    # Harami (inside bar)
    if c > o and c1 < o1 and c < o1 and o > c1:
        return "bullish_harami"
    if c < o and c1 > o1 and c > o1 and o < c1:
        return "bearish_harami"

    # Kicker: gap followed by opposite direction strong bar
    if idx >= 1:
        gap_up = min(o, c) > max(o1, c1)   # current bar completely above previous
        gap_down = max(o, c) < min(o1, c1) # current bar completely below previous
        if gap_up and c > o and c1 < o1:
            return "bullish_kicker"
        if gap_down and c < o and c1 > o1:
            return "bearish_kicker"

    # --------------------------------------------------------------
    # Three-bar patterns (need two previous bars)
    # --------------------------------------------------------------
    if idx < 2:
        return "none"
    prev2 = bars[idx-2]
    o2, c2 = prev2.open, prev2.close

    # Three white soldiers: three consecutive long bullish candles, each closing higher
    if c > o and c1 > o1 and c2 > o2:
        body_ratio1 = (c1 - o1) / (c1) if c1 != 0 else 0
        body_ratio2 = (c2 - o2) / (c2) if c2 != 0 else 0
        body_ratio0 = (c - o) / c if c != 0 else 0
        if body_ratio0 > 0.02 and body_ratio1 > 0.02 and body_ratio2 > 0.02 and c > c1 > c2:
            return "three_white_soldiers"

    # Three black crows: three consecutive long bearish candles, each closing lower
    if c < o and c1 < o1 and c2 < o2:
        body_ratio1 = (o1 - c1) / o1 if o1 != 0 else 0
        body_ratio2 = (o2 - c2) / o2 if o2 != 0 else 0
        body_ratio0 = (o - c) / o if o != 0 else 0
        if body_ratio0 > 0.02 and body_ratio1 > 0.02 and body_ratio2 > 0.02 and c < c1 < c2:
            return "three_black_crows"

    # Morning Star: bearish candle, then small body (doji or small), then bullish candle that closes above midpoint of first
    if c2 < o2 and abs(c1 - o1) < (c2 - o2)*0.3 and c > o and c > (o2 + c2)/2:
        return "morning_star"

    # Evening Star: bullish candle, then small body, then bearish candle that closes below midpoint of first
    if c2 > o2 and abs(c1 - o1) < (c2 - o2)*0.3 and c < o and c < (o2 + c2)/2:
        return "evening_star"

    # Abandoned Baby: gap followed by doji, then opposite gap (requires three bars)
    # Current bar must gap away from the doji bar
    if idx >= 2:
        # Bullish abandoned baby: down, doji, gap up
        if c2 < o2 and is_doji_pattern(prev) and c > o and min(o, c) > max(o1, c1):
            return "bullish_abandoned_baby"
        # Bearish abandoned baby: up, doji, gap down
        if c2 > o2 and is_doji_pattern(prev) and c < o and max(o, c) < min(o1, c1):
            return "bearish_abandoned_baby"

    return "none"

def is_doji_pattern(bar: OHLCV) -> bool:
    body = abs(bar.close - bar.open)
    total = bar.high - bar.low
    if total == 0:
        return False
    return body / total < 0.1