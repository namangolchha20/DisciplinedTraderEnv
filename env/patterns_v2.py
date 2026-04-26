import numpy as np
from typing import List, Tuple, Optional

# ----------------------------------------------------------------------
# Zig-Zag Swing Points (percentage-based)
# ----------------------------------------------------------------------

def zigzag_swings(prices: List[float], min_move_percent: float = 1.0) -> Tuple[List[int], List[int]]:
    """
    Extract significant swing highs and lows using a percentage threshold.
    Returns lists of indices where price reversed by at least `min_move_percent`.
    """
    if len(prices) < 3:
        return [], []
    high_idx = []
    low_idx = []
    direction = 0  # 1 = looking for high, -1 = looking for low
    last_price = prices[0]
    for i in range(1, len(prices)-1):
        current = prices[i]
        # Detect local extremes with price movement threshold
        if direction >= 0 and current > last_price * (1 + min_move_percent/100):
            # Up move, potential high
            if i > 1 and prices[i-1] <= last_price:
                high_idx.append(i-1)
                direction = -1
                last_price = prices[i-1]
        elif direction <= 0 and current < last_price * (1 - min_move_percent/100):
            # Down move, potential low
            if i > 1 and prices[i-1] >= last_price:
                low_idx.append(i-1)
                direction = 1
                last_price = prices[i-1]
    # Add the last point if it's an extreme
    if len(prices) > 1:
        if direction == -1 and prices[-2] < prices[-1]:
            high_idx.append(len(prices)-1)
        elif direction == 1 and prices[-2] > prices[-1]:
            low_idx.append(len(prices)-1)
    return high_idx, low_idx

# ----------------------------------------------------------------------
# Helper: Linear regression slope
# ----------------------------------------------------------------------

def slope(prices: List[float], indices: List[int]) -> float:
    """Slope of linear regression through points (x=index, y=price)."""
    if len(prices) < 2:
        return 0.0
    x = np.array(indices)
    y = np.array(prices)
    n = len(x)
    if n == 0:
        return 0.0
    slope_val = (n * np.sum(x*y) - np.sum(x)*np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
    return slope_val

# ----------------------------------------------------------------------
# Neckline for Head & Shoulders
# ----------------------------------------------------------------------

def neckline(prices: List[float], lows_idx: List[int], left_shoulder_idx: int, head_idx: int, right_shoulder_idx: int) -> Tuple[float, float]:
    """
    Compute neckline trendline connecting the troughs between left shoulder and head, and between head and right shoulder.
    Returns (slope, intercept) of the line.
    """
    # Find the lows between left shoulder and head, and between head and right shoulder
    low1_idx = max([i for i in lows_idx if left_shoulder_idx < i < head_idx], default=None)
    low2_idx = max([i for i in lows_idx if head_idx < i < right_shoulder_idx], default=None)
    if low1_idx is None or low2_idx is None:
        return 0.0, 0.0
    x1, y1 = low1_idx, prices[low1_idx]
    x2, y2 = low2_idx, prices[low2_idx]
    sl = (y2 - y1) / (x2 - x1) if x2 != x1 else 0.0
    intercept = y1 - sl * x1
    return sl, intercept

# ----------------------------------------------------------------------
# Pattern detection functions (each returns (pattern_name, confidence))
# ----------------------------------------------------------------------

def detect_head_shoulders(prices: List[float], highs_idx: List[int], lows_idx: List[int],
                          volumes: Optional[List[float]] = None,
                          min_symmetry_ratio: float = 0.7) -> Tuple[str, float]:
    """Regular Head & Shoulders (bearish reversal)."""
    if len(highs_idx) < 3:
        return "none", 0.0
    # Use the last three highs as potential left shoulder, head, right shoulder
    ls = highs_idx[-3]
    head = highs_idx[-2]
    rs = highs_idx[-1]
    # Price condition: head > left shoulder and head > right shoulder
    if not (prices[head] > prices[ls] and prices[head] > prices[rs]):
        return "none", 0.0
    # Symmetry: distance from left shoulder to head vs head to right shoulder
    time_sym = abs((head - ls) - (rs - head)) / max(rs - ls, 1)
    if time_sym > (1 - min_symmetry_ratio):
        return "none", 0.0
    # Volume (if available): left shoulder volume > head volume > right shoulder volume?
    if volumes and len(volumes) > rs:
        vol_ls = volumes[ls]
        vol_head = volumes[head]
        vol_rs = volumes[rs]
        volume_decay = (vol_ls > vol_head and vol_head > vol_rs)
        confidence = 0.8 if volume_decay else 0.5
    else:
        confidence = 0.6
    # Neckline break not required for detection, but we compute slope
    neck_slope, _ = neckline(prices, lows_idx, ls, head, rs)
    # If neckline slopes upward, it's more reliable
    if neck_slope > 0:
        confidence += 0.1
    return "head_and_shoulders", min(confidence, 1.0)

def detect_reverse_head_shoulders(prices: List[float], highs_idx: List[int], lows_idx: List[int],
                                  volumes: Optional[List[float]] = None,
                                  min_symmetry_ratio: float = 0.7) -> Tuple[str, float]:
    """Inverse Head & Shoulders (bullish reversal)."""
    if len(lows_idx) < 3:
        return "none", 0.0
    ls = lows_idx[-3]
    head = lows_idx[-2]
    rs = lows_idx[-1]
    if not (prices[head] < prices[ls] and prices[head] < prices[rs]):
        return "none", 0.0
    time_sym = abs((head - ls) - (rs - head)) / max(rs - ls, 1)
    if time_sym > (1 - min_symmetry_ratio):
        return "none", 0.0
    # Volume for inverse: high volume on head, lower on shoulders?
    confidence = 0.6
    return "reverse_head_and_shoulders", confidence

def detect_double_top(prices: List[float], highs_idx: List[int], lows_idx: List[int],
                      tolerance: float = 0.02) -> Tuple[str, float]:
    if len(highs_idx) < 2:
        return "none", 0.0
    idx1 = highs_idx[-2]
    idx2 = highs_idx[-1]
    diff = abs(prices[idx1] - prices[idx2]) / prices[idx1]
    if diff > tolerance:
        return "none", 0.0
    # Check for a trough in between
    trough = min(prices[idx1:idx2])
    if (prices[idx1] - trough) / prices[idx1] < 0.02:
        return "none", 0.0
    confidence = 0.7 - diff/tolerance
    return "double_top", max(0.3, min(1.0, confidence))

def detect_double_bottom(prices: List[float], lows_idx: List[int], tolerance: float = 0.02) -> Tuple[str, float]:
    if len(lows_idx) < 2:
        return "none", 0.0
    idx1 = lows_idx[-2]
    idx2 = lows_idx[-1]
    diff = abs(prices[idx1] - prices[idx2]) / prices[idx1]
    if diff > tolerance:
        return "none", 0.0
    peak = max(prices[idx1:idx2])
    if (peak - prices[idx1]) / prices[idx1] < 0.02:
        return "none", 0.0
    confidence = 0.7 - diff/tolerance
    return "double_bottom", max(0.3, min(1.0, confidence))

def detect_triple_top(prices: List[float], highs_idx: List[int], tolerance: float = 0.02) -> Tuple[str, float]:
    if len(highs_idx) < 3:
        return "none", 0.0
    idx1, idx2, idx3 = highs_idx[-3], highs_idx[-2], highs_idx[-1]
    peaks = [prices[idx1], prices[idx2], prices[idx3]]
    if max(peaks)/min(peaks) > 1 + tolerance:
        return "none", 0.0
    # Check drops
    drop1 = (prices[idx1] - min(prices[idx1:idx2])) / prices[idx1]
    drop2 = (prices[idx2] - min(prices[idx2:idx3])) / prices[idx2]
    if drop1 < 0.02 or drop2 < 0.02:
        return "none", 0.0
    return "triple_top", 0.8

def detect_triple_bottom(prices: List[float], lows_idx: List[int], tolerance: float = 0.02) -> Tuple[str, float]:
    if len(lows_idx) < 3:
        return "none", 0.0
    idx1, idx2, idx3 = lows_idx[-3], lows_idx[-2], lows_idx[-1]
    troughs = [prices[idx1], prices[idx2], prices[idx3]]
    if max(troughs)/min(troughs) > 1 + tolerance:
        return "none", 0.0
    rise1 = (max(prices[idx1:idx2]) - prices[idx1]) / prices[idx1]
    rise2 = (max(prices[idx2:idx3]) - prices[idx2]) / prices[idx2]
    if rise1 < 0.02 or rise2 < 0.02:
        return "none", 0.0
    return "triple_bottom", 0.8

def detect_wedge(prices: List[float], highs_idx: List[int], lows_idx: List[int],
                 lookback: int = 5) -> Tuple[str, float]:
    """Rising or falling wedge using recent swings."""
    if len(highs_idx) < lookback or len(lows_idx) < lookback:
        return "none", 0.0
    recent_highs = [prices[i] for i in highs_idx[-lookback:]]
    recent_lows = [prices[i] for i in lows_idx[-lookback:]]
    high_indices = highs_idx[-lookback:]
    low_indices = lows_idx[-lookback:]
    high_slope = slope(recent_highs, high_indices)
    low_slope = slope(recent_lows, low_indices)
    if high_slope > 0 and low_slope > 0 and high_slope < low_slope:
        return "rising_wedge", 0.7
    if high_slope < 0 and low_slope < 0 and high_slope > low_slope:
        return "falling_wedge", 0.7
    return "none", 0.0

def detect_expanding_triangle(prices: List[float], highs_idx: List[int], lows_idx: List[int],
                              lookback: int = 4) -> Tuple[str, float]:
    if len(highs_idx) < lookback or len(lows_idx) < lookback:
        return "none", 0.0
    recent_highs = [prices[i] for i in highs_idx[-lookback:]]
    recent_lows = [prices[i] for i in lows_idx[-lookback:]]
    high_indices = highs_idx[-lookback:]
    low_indices = lows_idx[-lookback:]
    high_slope = slope(recent_highs, high_indices)
    low_slope = slope(recent_lows, low_indices)
    if high_slope > 0 and low_slope < 0:
        return "bullish_expanding_triangle", 0.6
    if high_slope < 0 and low_slope > 0:
        return "bearish_expanding_triangle", 0.6
    return "none", 0.0

def detect_rectangle(prices: List[float], highs_idx: List[int], lows_idx: List[int],
                     tolerance: float = 0.02, lookback=3) -> Tuple[str, float]:
    if len(highs_idx) < lookback or len(lows_idx) < lookback:
        return "none", 0.0
    recent_highs = [prices[i] for i in highs_idx[-lookback:]]
    recent_lows = [prices[i] for i in lows_idx[-lookback:]]
    high_range = max(recent_highs) - min(recent_highs)
    low_range = max(recent_lows) - min(recent_lows)
    avg_price = np.mean(recent_highs + recent_lows)
    if high_range / avg_price < tolerance and low_range / avg_price < tolerance:
        current_price = prices[-1]
        if current_price > max(recent_highs) * 0.99:
            return "bullish_rectangle", 0.7
        elif current_price < min(recent_lows) * 1.01:
            return "bearish_rectangle", 0.7
    return "none", 0.0

def detect_flag_pennant(prices: List[float], lookback: int = 20,
                        min_sharp_move: float = 0.04,
                        max_consolidation_range: float = 0.02) -> Tuple[str, float]:
    """Detect flag or pennant based on sharp move + consolidation."""
    if len(prices) < lookback:
        return "none", 0.0
    segment = prices[-lookback:]
    first_half = segment[:lookback//2]
    second_half = segment[lookback//2:]
    move = abs(first_half[-1] - first_half[0]) / first_half[0]
    if move < min_sharp_move:
        return "none", 0.0
    # Consolidation range
    con_range = (max(second_half) - min(second_half)) / np.mean(second_half)
    if con_range > max_consolidation_range:
        return "none", 0.0
    # Determine type: flag = parallel trendlines (simple check), pennant = converging
    # We'll just classify as flag if range is small
    direction = "bullish" if segment[-1] > segment[0] else "bearish"
    return f"{direction}_flag", 0.65  # we can later differentiate pennant

def detect_symmetrical_triangle(prices: List[float], highs_idx: List[int], lows_idx: List[int],
                                lookback: int = 5) -> Tuple[str, float]:
    if len(highs_idx) < lookback or len(lows_idx) < lookback:
        return "none", 0.0
    recent_highs = [prices[i] for i in highs_idx[-lookback:]]
    recent_lows = [prices[i] for i in lows_idx[-lookback:]]
    high_indices = highs_idx[-lookback:]
    low_indices = lows_idx[-lookback:]
    high_slope = slope(recent_highs, high_indices)
    low_slope = slope(recent_lows, low_indices)
    if high_slope < 0 and low_slope > 0:
        # converging
        return "bullish_symmetrical_triangle", 0.7
    return "none", 0.0

def detect_cup_handle(prices: List[float], lows_idx: List[int], tolerance: float = 0.02) -> Tuple[str, float]:
    """
    Simplified cup and handle: U-shaped bottom followed by a small dip (handle).
    Requires at least 4 swing lows.
    """
    if len(lows_idx) < 4:
        return "none", 0.0
    # Use last 4 lows as candidate cup formation: L1, L2, L3, L4
    l1, l2, l3, l4 = lows_idx[-4], lows_idx[-3], lows_idx[-2], lows_idx[-1]
    # Cup shape: L1 > L2 < L3 and L3 > L4 (or L4 slightly lower than L3 for handle)
    if not (prices[l1] > prices[l2] and prices[l2] < prices[l3]):
        return "none", 0.0
    # Handle: L4 is near L3 but not higher than L3
    if not (prices[l4] <= prices[l3] * (1 + tolerance)):
        return "none", 0.0
    # Check that the cup depth is significant
    cup_depth = (prices[l1] - prices[l2]) / prices[l1]
    if cup_depth < 0.05:
        return "none", 0.0
    return "cup_and_handle", 0.8

# ----------------------------------------------------------------------
# Main detection function (returns pattern name and confidence)
# ----------------------------------------------------------------------

def detect_chart_pattern_v2(prices: List[float], volumes: Optional[List[float]] = None) -> Tuple[str, float]:
    """
    Detect chart patterns using zig-zag swing points and robust criteria.
    Returns (pattern_name, confidence_score) where confidence in [0,1].
    """
    if len(prices) < 50:
        return "none", 0.0
    # Get zig-zag swings (1% min move works for most patterns)
    highs_idx, lows_idx = zigzag_swings(prices, min_move_percent=1.0)
    if len(highs_idx) < 3 and len(lows_idx) < 3:
        return "none", 0.0
    # Order matters: detect more specific patterns first
    pattern, conf = detect_head_shoulders(prices, highs_idx, lows_idx, volumes)
    if pattern != "none":
        return pattern, conf
    pattern, conf = detect_reverse_head_shoulders(prices, highs_idx, lows_idx, volumes)
    if pattern != "none":
        return pattern, conf
    pattern, conf = detect_cup_handle(prices, lows_idx)
    if pattern != "none":
        return pattern, conf
    pattern, conf = detect_triple_top(prices, highs_idx)
    if pattern != "none":
        return pattern, conf
    pattern, conf = detect_triple_bottom(prices, lows_idx)
    if pattern != "none":
        return pattern, conf
    pattern, conf = detect_double_top(prices, highs_idx, lows_idx)
    if pattern != "none":
        return pattern, conf
    pattern, conf = detect_double_bottom(prices, lows_idx)
    if pattern != "none":
        return pattern, conf
    pattern, conf = detect_symmetrical_triangle(prices, highs_idx, lows_idx)
    if pattern != "none":
        return pattern, conf
    pattern, conf = detect_wedge(prices, highs_idx, lows_idx)
    if pattern != "none":
        return pattern, conf
    pattern, conf = detect_expanding_triangle(prices, highs_idx, lows_idx)
    if pattern != "none":
        return pattern, conf
    pattern, conf = detect_rectangle(prices, highs_idx, lows_idx)
    if pattern != "none":
        return pattern, conf
    pattern, conf = detect_flag_pennant(prices)
    if pattern != "none":
        return pattern, conf
    return "none", 0.0