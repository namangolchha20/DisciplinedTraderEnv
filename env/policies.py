"""Reference policies: baselines for evaluation and a disciplined demo bot
that powers the autopilot mode in the web terminal."""

import random
from .models import Observation, Action
from .environment import BULLISH_CANDLES, BEARISH_CANDLES, BULLISH_CHARTS, BEARISH_CHARTS


def random_policy(obs: Observation) -> Action:
    chosen = random.choice(["open_long", "open_short", "close_position", "do_nothing"])
    if chosen in ("open_long", "open_short"):
        return Action(action_type=chosen, amount_shares=random.randint(5, 20))
    return Action(action_type=chosen, amount_shares=0)


def sma_crossover_policy(obs: Observation) -> Action:
    """Classic trend baseline: 15m close crossing its SMA20. (The original
    version compared against a `sma50` field that never existed, so it
    silently never traded.) Uses the 15m frame so short episodes still have
    enough candles for the SMA to be meaningful."""
    tf = obs.tf_15m
    price = tf.ohlcv.close

    if obs.position_shares == 0:
        if obs.time_since_last_trade < 10:
            return Action(action_type="do_nothing", amount_shares=0)
        if price > tf.moving_average * 1.001:
            return Action(action_type="open_long", amount_shares=20)
        if price < tf.moving_average * 0.999:
            return Action(action_type="open_short", amount_shares=20)
    elif obs.position_shares > 0 and price < tf.moving_average:
        return Action(action_type="close_position", amount_shares=0)
    elif obs.position_shares < 0 and price > tf.moving_average:
        return Action(action_type="close_position", amount_shares=0)
    return Action(action_type="do_nothing", amount_shares=0)


# ----------------------------------------------------------------------
# Multi-timeframe confluence
# ----------------------------------------------------------------------
# Every timeframe casts weighted votes; higher timeframes carry more weight
# because their signals are slower and more reliable. The bot only acts when
# total confirmation crosses a threshold AND clearly beats the opposite side.
TF_WEIGHTS = (
    ("tf_1m", 1.0),
    ("tf_5m", 1.25),
    ("tf_15m", 1.5),
    ("tf_1h", 1.75),
    ("tf_1d", 2.0),
)
# Max vote per timeframe = 1 (SuperTrend) + 0.5 (SMA) + 0.5 (RSI)
#                        + 1 (chart pattern) + 0.5 (candle) = 3.5
CONFLUENCE_MAX = 3.5 * sum(w for _, w in TF_WEIGHTS) + 2.0  # +2 regime

ENTRY_SCORE = 7.0     # minimum absolute confirmation to enter
ENTRY_MARGIN = 3.5    # how much the winning side must beat the other side by
EXIT_MARGIN = 3.0     # opposite-side dominance that forces an exit
MIN_HOLD_BARS = 8     # no voluntary exits before this (stop-loss still protects)


def _vol_pct(t) -> float:
    """Per-bar volatility (std as a fraction of price) from Bollinger width."""
    if t.moving_average > 0 and t.bb_upper > t.bb_lower:
        return (t.bb_upper - t.bb_lower) / (4 * t.moving_average)
    return 0.005


def _trail_gap(obs: Observation) -> float:
    """Volatility-adaptive trailing distance: wide enough to survive normal
    noise (≈2.5 bars of std), tight enough to lock in trend profits."""
    return min(0.05, max(0.01, 2.5 * _vol_pct(obs.tf_1m)))


def confluence_score(obs: Observation):
    """Aggregate weighted bull/bear evidence across all five timeframes.

    Returns (bull, bear, breakdown) where breakdown maps each timeframe to
    its net vote (positive = bullish) for display/debugging.
    """
    bull, bear = 0.0, 0.0
    breakdown = {}
    for name, w in TF_WEIGHTS:
        t = getattr(obs, name)
        price = t.ohlcv.close
        b = s = 0.0

        # Trend votes
        if t.super_trend_direction == 1:
            b += 1.0
        elif t.super_trend_direction == -1:
            s += 1.0
        if price > t.moving_average * 1.001:
            b += 0.5
        elif price < t.moving_average * 0.999:
            s += 0.5

        # Momentum extremes (mean reversion votes)
        if t.rsi <= 30:
            b += 0.5
        elif t.rsi >= 70:
            s += 0.5

        # Pattern votes
        if t.chart_pattern in BULLISH_CHARTS:
            b += 1.0
        elif t.chart_pattern in BEARISH_CHARTS:
            s += 1.0
        if t.candlestick_pattern in BULLISH_CANDLES:
            b += 0.5
        elif t.candlestick_pattern in BEARISH_CANDLES:
            s += 0.5

        bull += b * w
        bear += s * w
        breakdown[name.replace("tf_", "")] = round((b - s) * w, 2)

    if obs.market_regime == "uptrend":
        bull += 2.0
    elif obs.market_regime == "downtrend":
        bear += 2.0

    return round(bull, 2), round(bear, 2), breakdown


def manage_open_position(obs: Observation) -> Action:
    """Trail stops and confluence exits for an open position."""
    if obs.position_shares == 0:
        return Action(action_type="do_nothing", amount_shares=0)

    price = obs.tf_1m.ohlcv.close
    bull, bear, _ = confluence_score(obs)
    entry = obs.position_entry_price
    held = obs.time_since_last_trade
    stop_price = price * (1 + obs.stop_loss_distance) if obs.stop_loss_distance else None
    gap = _trail_gap(obs)

    if held <= 2 and obs.stop_loss_distance and abs(obs.stop_loss_distance) < gap * 0.8:
        return Action(action_type="set_stop_loss", stop_loss_percent=gap)

    if obs.position_shares > 0:
        unreal = price / entry - 1 if entry > 0 else 0.0
        if unreal >= gap:
            proposed = price * (1 - gap)
            if stop_price is None or proposed > stop_price * 1.002:
                return Action(action_type="set_stop_loss", stop_loss_percent=gap)
        if held >= MIN_HOLD_BARS and (bear - bull) >= EXIT_MARGIN:
            return Action(action_type="close_position", amount_shares=0)
    else:
        unreal = entry / price - 1 if price > 0 else 0.0
        if unreal >= gap:
            proposed = price * (1 + gap)
            if stop_price is None or proposed < stop_price * 0.998:
                return Action(action_type="set_stop_loss", stop_loss_percent=gap)
        if held >= MIN_HOLD_BARS and (bull - bear) >= EXIT_MARGIN:
            return Action(action_type="close_position", amount_shares=0)
    return Action(action_type="do_nothing", amount_shares=0)


def llm_position_overlay(obs: Observation, llm_action: Action) -> Action:
    """Let the LLM pick entries/exits; overlay bot risk management when it waits."""
    if obs.position_shares == 0:
        return llm_action
    if llm_action.action_type == "close_position":
        return llm_action
    managed = manage_open_position(obs)
    if managed.action_type != "do_nothing":
        return managed
    return llm_action


def disciplined_bot(obs: Observation) -> Action:
    """Showcase policy embodying the rules the environment rewards: act only
    on multi-timeframe confluence, respect a cooldown between trades, take
    profits, and exit when the higher-timeframe picture flips against you."""
    price = obs.tf_1m.ohlcv.close
    bull, bear, _ = confluence_score(obs)

    if obs.position_shares != 0:
        return manage_open_position(obs)

    # ---- Entry: respect the cooldown (no overtrading) ----
    cooldown = max(25, obs.max_bars // 25)
    if obs.time_since_last_trade < cooldown:
        return Action(action_type="do_nothing", amount_shares=0)

    shares = int(obs.cash * 0.25 / price) if price > 0 else 0
    if shares <= 0:
        return Action(action_type="do_nothing", amount_shares=0)

    # Enter only with strong absolute confirmation AND clear directional edge,
    # never against the daily regime, never into exhausted momentum.
    if bull >= ENTRY_SCORE and (bull - bear) >= ENTRY_MARGIN \
            and obs.market_regime != "downtrend" and obs.tf_15m.rsi < 70:
        return Action(action_type="open_long", amount_shares=shares)
    if bear >= ENTRY_SCORE and (bear - bull) >= ENTRY_MARGIN \
            and obs.market_regime != "uptrend" and obs.tf_15m.rsi > 30:
        return Action(action_type="open_short", amount_shares=shares)
    return Action(action_type="do_nothing", amount_shares=0)


POLICIES = {
    "bot": disciplined_bot,
    "random": random_policy,
    "sma": sma_crossover_policy,
}
