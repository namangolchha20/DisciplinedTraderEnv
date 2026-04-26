from pydantic import BaseModel
from typing import List, Optional

class OHLCV(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float

class TimeframeData(BaseModel):
    ohlcv: OHLCV
    rsi: float
    moving_average: float
    bb_upper: float
    bb_middle: float               # same as moving_average
    bb_lower: float
    super_trend_value: float
    super_trend_direction: int     # 1 = uptrend, -1 = downtrend, 0 = none
    volume_index: float            # ratio of current volume to average volume (e.g., 1.5 = 50% above avg)
    support: float
    resistance: float
    candlestick_pattern: str
    chart_pattern: str

class Observation(BaseModel):
    # Account & position info
    cash: float                       # available cash balance
    account_value: float              # total account value (cash + position value)
    position_shares: int              # current number of shares held (positive = long, negative = short)
    position_entry_price: float       # average entry price of current position
    risk_usage: float                 # position value / total account (capped at 0.3)
    stop_loss_distance: float         # current trailing stop loss as % from entry (e.g., 0.02 = 2%)
    time_since_last_trade: int        # bars elapsed since last trade (for overtrade detection)

    # Multi-timeframe data (each timeframe has its own indicators & patterns)
    tf_1m: TimeframeData
    tf_5m: TimeframeData
    tf_15m: TimeframeData
    tf_1h: TimeframeData
    tf_1d: TimeframeData

    # Market regime & episode state
    market_regime: str                # "uptrend", "downtrend", or "sideways"
    current_bar: int                  # current step index (0..max_bars-1) -> current candle index 
    max_bars: int                     # total number of bars in episode -> total candles in the episode

class Action(BaseModel):
    action_type: str                  # "open_long", "open_short", "close_position", "set_stop_loss", "do_nothing"
    amount_shares: Optional[int] = 0      # number of shares to buy/sell when opening a position -> will be calculated based on the available cash balance and risk_budget
    stop_loss_percent: Optional[float] = None  # fixed stop loss as percentage from entry (e.g., 0.02 = 2%). Not automatic trailing; agent can adjust by calling again.

class Info(BaseModel):
    profit: float                     # total profit from all trades -> final portfolio value - initial capital
    drawdown: float                   # maximum drawdown from peak portfolio value (percentage)
    win_rate: float                   # percentage of winning trades
    total_trades: int

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Info