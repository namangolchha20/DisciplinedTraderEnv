import random
import numpy as np
from openenv.core.env_server import Environment
from .models import Observation, Action, StepResult, Info, TimeframeData
from .data_generator import generate_synthetic_ohlcv, resample_ohlcv
from .indicators import (
    compute_rsi, compute_sma, compute_bollinger_bands,
    compute_super_trend, compute_volume_index,
    detect_support_resistance, detect_candlestick_pattern
)
from .patterns_v2 import detect_chart_pattern_v2 as detect_chart_pattern

INITIAL_CAPITAL = 10_000.0
# Indicators / patterns only need a bounded lookback. Capping the window keeps
# per-step cost O(1) instead of O(n), which makes long episodes ~20x faster.
INDICATOR_WINDOW = 240

BULLISH_CANDLES = {
    "hammer", "inverted_hammer", "bullish_engulfing", "bullish_harami",
    "bullish_kicker", "morning_star", "three_white_soldiers",
    "bullish_marubozu", "dragonfly_doji", "bullish_abandoned_baby",
}
BEARISH_CANDLES = {
    "shooting_star", "hanging_man", "bearish_engulfing", "bearish_harami",
    "bearish_kicker", "evening_star", "three_black_crows",
    "bearish_marubozu", "gravestone_doji", "bearish_abandoned_baby",
}
BULLISH_CHARTS = {
    "double_bottom", "triple_bottom", "cup_and_handle",
    "reverse_head_and_shoulders", "falling_wedge", "bullish_flag",
    "bullish_rectangle", "bullish_symmetrical_triangle",
}
BEARISH_CHARTS = {
    "head_and_shoulders", "double_top", "triple_top",
    "rising_wedge", "bearish_flag", "bearish_rectangle",
}


class DisciplinedTraderEnv(Environment):
    def __init__(self):
        self.max_bars = 5000
        self._rng = None
        self.cash = INITIAL_CAPITAL
        self.position_shares = 0
        self.entry_price = 0.0
        self.stop_loss = None
        self.last_trade_bar = 0
        self.trades = []
        self.ohlcv_1m = []
        self.ohlcv_5m = []
        self.ohlcv_15m = []
        self.ohlcv_1h = []
        self.ohlcv_1d = []
        self.current_step = 0
        self.task = None
        self.peak_value = INITIAL_CAPITAL
        self.max_drawdown = 0.0
        self.equity_curve = []
        self._cached_obs = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self, task_id: str = None, seed: int = None) -> Observation:
        if seed is not None:
            self._rng = random.Random(seed)
            np.random.seed(seed)
        else:
            self._rng = random.Random()
        self.current_step = 0
        self.cash = INITIAL_CAPITAL
        self.position_shares = 0
        self.entry_price = 0.0
        self.stop_loss = None
        self.last_trade_bar = 0
        self.trades = []
        self.peak_value = INITIAL_CAPITAL
        self.max_drawdown = 0.0
        self.equity_curve = [INITIAL_CAPITAL]
        self.task = task_id if task_id else "easy"

        if self.task == "easy":
            volatility, trend = 0.005, 0.0002
            bars = 500
        elif self.task == "medium":
            volatility, trend = 0.01, 0.0001
            bars = 1500
        else:
            volatility, trend = 0.02, 0.0
            bars = 3000
        self.max_bars = bars
        raw_1m = generate_synthetic_ohlcv(start_price=100, num_bars=bars, volatility=volatility, trend=trend)
        self.ohlcv_1m = raw_1m
        self.ohlcv_5m = resample_ohlcv(raw_1m, 5)
        self.ohlcv_15m = resample_ohlcv(raw_1m, 15)
        self.ohlcv_1h = resample_ohlcv(raw_1m, 60)
        self.ohlcv_1d = resample_ohlcv(raw_1m, 390)

        obs = self._get_observation()
        self._cached_obs = obs
        return obs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _current_price(self) -> float:
        idx = min(self.current_step, len(self.ohlcv_1m) - 1)
        return self.ohlcv_1m[idx].close

    def account_value(self) -> float:
        """Mark-to-market equity: cash + open position value."""
        return self.cash + self.position_shares * self._current_price()

    def _close_position(self, price: float) -> float:
        """Liquidate the open position at `price`. Returns realised profit."""
        # Works for both directions: longs sell (cash += s*p), shorts buy to
        # cover (cash += (-s)*p), and profit = shares * (exit - entry) holds
        # with signed shares.
        self.cash += self.position_shares * price
        profit = self.position_shares * (price - self.entry_price)
        self.trades.append((self.last_trade_bar, self.current_step, profit))
        self.position_shares = 0
        self.entry_price = 0.0
        self.stop_loss = None
        return profit

    def _get_observation(self) -> Observation:
        idx_1m = min(self.current_step, len(self.ohlcv_1m) - 1)
        idx_5m = idx_1m // 5
        idx_15m = idx_1m // 15
        idx_1h = idx_1m // 60
        idx_1d = idx_1m // 390

        def make_tf_data(ohlcv_list, idx):
            idx = max(0, min(idx, len(ohlcv_list) - 1))
            bar = ohlcv_list[idx]
            start = max(0, idx + 1 - INDICATOR_WINDOW)
            window = ohlcv_list[start:idx + 1]
            prices = [b.close for b in window]
            highs = [b.high for b in window]
            lows = [b.low for b in window]
            volumes = [b.volume for b in window]

            rsi = compute_rsi(prices)
            sma20 = compute_sma(prices, 20)
            bb_upper, bb_middle, bb_lower = compute_bollinger_bands(prices, 20, 2)
            super_val, super_dir = compute_super_trend(highs, lows, prices, 10, 3)
            vol_idx = compute_volume_index(volumes, 20)

            sup, res = detect_support_resistance(highs, lows, 20)
            candle_pattern = detect_candlestick_pattern(ohlcv_list, idx)
            chart_pattern, _ = detect_chart_pattern(prices, volumes)

            return TimeframeData(
                ohlcv=bar,
                rsi=rsi,
                moving_average=sma20,
                bb_upper=bb_upper,
                bb_middle=bb_middle,
                bb_lower=bb_lower,
                super_trend_value=super_val,
                super_trend_direction=super_dir,
                volume_index=vol_idx,
                support=sup,
                resistance=res,
                candlestick_pattern=candle_pattern,
                chart_pattern=chart_pattern
            )

        tf1 = make_tf_data(self.ohlcv_1m, idx_1m)
        tf5 = make_tf_data(self.ohlcv_5m, idx_5m)
        tf15 = make_tf_data(self.ohlcv_15m, idx_15m)
        tf1h = make_tf_data(self.ohlcv_1h, idx_1h)
        tf1d = make_tf_data(self.ohlcv_1d, idx_1d)

        account_value = self.cash + self.position_shares * tf1.ohlcv.close
        if account_value <= 0:
            account_value = 0.01
        risk_usage = (self.position_shares * tf1.ohlcv.close) / account_value
        stop_dist = (self.stop_loss / tf1.ohlcv.close - 1) if self.stop_loss else 0.0

        if account_value > self.peak_value:
            self.peak_value = account_value

        if tf1d.moving_average > tf1d.ohlcv.close * 1.01:
            regime = "downtrend"
        elif tf1d.moving_average < tf1d.ohlcv.close * 0.99:
            regime = "uptrend"
        else:
            regime = "sideways"

        return Observation(
            cash=self.cash,
            account_value=account_value,
            position_shares=self.position_shares,
            position_entry_price=self.entry_price,
            risk_usage=risk_usage,
            stop_loss_distance=stop_dist,
            time_since_last_trade=self.current_step - self.last_trade_bar,
            tf_1m=tf1,
            tf_5m=tf5,
            tf_15m=tf15,
            tf_1h=tf1h,
            tf_1d=tf1d,
            market_regime=regime,
            current_bar=self.current_step,
            max_bars=self.max_bars
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: Action) -> StepResult:
        # The observation returned by the previous step() describes the same
        # bar we are acting on now, so reuse it instead of recomputing.
        obs_before = self._cached_obs if self._cached_obs is not None else self._get_observation()
        current_price = obs_before.tf_1m.ohlcv.close
        reward = 0.0

        prev_candle_pattern = obs_before.tf_1m.candlestick_pattern
        prev_chart_pattern = obs_before.tf_1m.chart_pattern

        # ---------------- Action execution ----------------
        if action.action_type == "open_long":
            if self.position_shares != 0:
                reward -= 0.1
            else:
                shares = min(action.amount_shares or 0, int(self.cash * 0.3 / current_price))
                if shares > 0:
                    self.position_shares = shares
                    self.entry_price = current_price
                    self.cash -= shares * current_price
                    self.last_trade_bar = self.current_step
                    reward -= 0.02
                    self.stop_loss = current_price * 0.98

        elif action.action_type == "open_short":
            if self.position_shares != 0:
                reward -= 0.1
            else:
                shares = min(action.amount_shares or 0, int(self.cash * 0.3 / current_price))
                if shares > 0:
                    self.position_shares = -shares
                    self.entry_price = current_price
                    self.cash += shares * current_price
                    self.last_trade_bar = self.current_step
                    reward -= 0.02
                    self.stop_loss = current_price * 1.02

        elif action.action_type == "close_position":
            if self.position_shares != 0:
                profit = self._close_position(current_price)
                reward += profit / 1000.0

        elif action.action_type == "set_stop_loss":
            if action.stop_loss_percent:
                if self.position_shares > 0:
                    self.stop_loss = current_price * (1 - action.stop_loss_percent)
                elif self.position_shares < 0:
                    self.stop_loss = current_price * (1 + action.stop_loss_percent)
                reward += 0.01

        # ---------------- Stop loss check ----------------
        if self.position_shares != 0 and self.stop_loss:
            stopped = (self.position_shares > 0 and current_price <= self.stop_loss) or \
                      (self.position_shares < 0 and current_price >= self.stop_loss)
            if stopped:
                profit = self._close_position(current_price)
                reward += profit / 1000.0

        # ---------------- Pattern alignment bonus ----------------
        if action.action_type == "open_long" and prev_candle_pattern in BULLISH_CANDLES:
            reward += 0.05
        if action.action_type == "open_short" and prev_candle_pattern in BEARISH_CANDLES:
            reward += 0.05
        if action.action_type == "open_long" and prev_chart_pattern in BULLISH_CHARTS:
            reward += 0.1
        if action.action_type == "open_short" and prev_chart_pattern in BEARISH_CHARTS:
            reward += 0.1

        # ---------------- Discipline shaping ----------------
        # Let winners run: small bonus for holding a profitable position.
        if self.position_shares != 0 and self.current_step - self.last_trade_bar > 5:
            unrealized = self.position_shares * (current_price - self.entry_price)
            if unrealized > 0:
                reward += 0.005

        regime = obs_before.market_regime
        if (regime == "uptrend" and self.position_shares > 0) or (regime == "downtrend" and self.position_shares < 0):
            reward += 0.01
        elif self.position_shares != 0:
            reward -= 0.005

        # Naked positions (no stop loss) are undisciplined.
        if self.position_shares != 0 and self.stop_loss is None:
            reward -= 0.02

        # Time cost of capital.
        reward -= 0.001

        # Revenge trading: re-entering within 20 bars of a significant LOSS.
        if self.trades and action.action_type in ("open_long", "open_short"):
            bars_since_exit = self.current_step - self.trades[-1][1]
            last_profit = self.trades[-1][2]
            if bars_since_exit < 20 and last_profit < -0.02 * max(self.cash, 1.0):
                reward -= 0.1

        # ---------------- Advance time ----------------
        self.current_step += 1
        done = self.current_step >= self.max_bars

        new_price = self._current_price()

        # Forced liquidation at episode end (actually move the cash!).
        if done and self.position_shares != 0:
            profit = self._close_position(new_price)
            reward += profit / 1000.0

        # ---------------- Equity / drawdown tracking ----------------
        equity = self.cash + self.position_shares * new_price
        self.equity_curve.append(equity)
        if equity > self.peak_value:
            self.peak_value = equity
        if self.peak_value > 0:
            dd = (self.peak_value - equity) / self.peak_value
            if dd > self.max_drawdown:
                self.max_drawdown = dd

        if done:
            if len(self.trades) > 1:
                returns = [p for _, _, p in self.trades]
                sharpe = np.mean(returns) / (np.std(returns) + 1e-9)
                reward += sharpe * 0.1
            reward -= self.max_drawdown * 0.2

        total_profit = sum(p for _, _, p in self.trades)
        wins = sum(1 for _, _, p in self.trades if p > 0)
        win_rate = wins / len(self.trades) if self.trades else 0.0
        info = Info(profit=total_profit, drawdown=self.max_drawdown, win_rate=win_rate, total_trades=len(self.trades))

        obs = self._get_observation()
        self._cached_obs = obs
        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def state(self) -> dict:
        return {
            "cash": self.cash,
            "position": self.position_shares,
            "step": self.current_step,
            "equity": self.account_value(),
            "max_drawdown": self.max_drawdown,
            "trades": len(self.trades),
        }
