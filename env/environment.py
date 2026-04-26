import random
import numpy as np
from typing import List
from openenv.core.env_server import Environment
from .models import Observation, Action, StepResult, Info, TimeframeData
from .data_generator import generate_synthetic_ohlcv, resample_ohlcv
from .indicators import (
    compute_rsi, compute_sma, compute_bollinger_bands,
    compute_super_trend, compute_volume_index,
    detect_support_resistance, detect_candlestick_pattern
)
from .patterns_v2 import detect_chart_pattern_v2 as detect_chart_pattern

class DisciplinedTraderEnv(Environment):
    def __init__(self):
        self.max_bars = 5000
        self._rng = None
        self.cash = 10000.0
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
        self.prev_pattern = None
        self.peak_value = 10000.0

    def reset(self, task_id: str = None, seed: int = None) -> Observation:
        if seed is not None:
            self._rng = random.Random(seed)
            np.random.seed(seed)
        else:
            self._rng = random.Random()
        self.current_step = 0
        self.cash = 10000.0
        self.position_shares = 0
        self.entry_price = 0.0
        self.stop_loss = None
        self.last_trade_bar = 0
        self.trades = []
        self.prev_pattern = "none"
        self.peak_value = 10000.0
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

        return self._get_observation()

    def _get_observation(self) -> Observation:
        idx_1m = self.current_step
        idx_5m = idx_1m // 5
        idx_15m = idx_1m // 15
        idx_1h = idx_1m // 60
        idx_1d = idx_1m // 390

        def make_tf_data(ohlcv_list, idx):
            if idx >= len(ohlcv_list):
                idx = len(ohlcv_list) - 1
            if idx < 0:
                idx = 0
            bar = ohlcv_list[idx]
            prices = [b.close for b in ohlcv_list[:idx+1]]
            highs = [b.high for b in ohlcv_list[:idx+1]]
            lows = [b.low for b in ohlcv_list[:idx+1]]
            volumes = [b.volume for b in ohlcv_list[:idx+1]]

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

    def step(self, action: Action) -> StepResult:
        obs_before = self._get_observation()
        current_price = obs_before.tf_1m.ohlcv.close
        reward = 0.0

        prev_candle_pattern = obs_before.tf_1m.candlestick_pattern
        prev_chart_pattern = obs_before.tf_1m.chart_pattern

        # Action execution (unchanged)
        if action.action_type == "open_long":
            if self.position_shares != 0:
                reward -= 0.1
            else:
                shares = min(action.amount_shares, int(self.cash * 0.3 / current_price))
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
                shares = min(action.amount_shares, int(self.cash * 0.3 / current_price))
                if shares > 0:
                    self.position_shares = -shares
                    self.entry_price = current_price
                    self.cash += shares * current_price
                    self.last_trade_bar = self.current_step
                    reward -= 0.02
                    self.stop_loss = current_price * 1.02

        elif action.action_type == "close_position":
            if self.position_shares != 0:
                close_price = current_price
                if self.position_shares > 0:
                    self.cash += self.position_shares * close_price
                    profit = self.position_shares * (close_price - self.entry_price)
                else:
                    self.cash += self.position_shares * close_price
                    profit = -self.position_shares * (self.entry_price - close_price)
                self.trades.append((self.last_trade_bar, self.current_step, profit))
                self.position_shares = 0
                self.entry_price = 0.0
                self.stop_loss = None
                reward += profit / 1000.0
                if profit / (self.cash - profit) > 0.1:
                    reward -= 0.1

        elif action.action_type == "set_stop_loss":
            if action.stop_loss_percent:
                if self.position_shares > 0:
                    self.stop_loss = current_price * (1 - action.stop_loss_percent)
                elif self.position_shares < 0:
                    self.stop_loss = current_price * (1 + action.stop_loss_percent)
                reward += 0.01

        # Stop loss check
        if self.position_shares != 0 and self.stop_loss:
            if (self.position_shares > 0 and current_price <= self.stop_loss) or (self.position_shares < 0 and current_price >= self.stop_loss):
                if self.position_shares > 0:
                    self.cash += self.position_shares * current_price
                    profit = self.position_shares * (current_price - self.entry_price)
                else:
                    self.cash += self.position_shares * current_price
                    profit = -self.position_shares * (self.entry_price - current_price)
                self.trades.append((self.last_trade_bar, self.current_step, profit))
                self.position_shares = 0
                self.entry_price = 0.0
                self.stop_loss = None
                reward += profit / 1000.0
                if profit / (self.cash - profit) > 0.1:
                    reward -= 0.1

        # Pattern bonus
        bullish_patterns = ["hammer", "bullish_engulfing", "morning_star", "cup_and_handle", "double_bottom"]
        bearish_patterns = ["shooting_star", "bearish_engulfing", "evening_star", "head_and_shoulders", "double_top"]
        if action.action_type == "open_long" and prev_candle_pattern in bullish_patterns:
            reward += 0.05
        if action.action_type == "open_short" and prev_candle_pattern in bearish_patterns:
            reward += 0.05
        if action.action_type == "open_short" and prev_chart_pattern in ["head_and_shoulders", "double_top"]:
            reward += 0.1
        if action.action_type == "open_long" and prev_chart_pattern in ["double_bottom", "cup_and_handle"]:
            reward += 0.1

        # Additional penalties / bonuses
        if self.position_shares != 0 and self.current_step - self.last_trade_bar > 5:
            unrealized = self.position_shares * (current_price - self.entry_price) if self.position_shares > 0 else -self.position_shares * (self.entry_price - current_price)
            if unrealized > 0:
                reward += 0.005

        regime = obs_before.market_regime
        if (regime == "uptrend" and self.position_shares > 0) or (regime == "downtrend" and self.position_shares < 0):
            reward += 0.01
        elif self.position_shares != 0:
            reward -= 0.005

        if self.position_shares != 0 and self.stop_loss is None:
            reward -= 0.02

        reward -= 0.001

        if self.trades and (self.current_step - self.trades[-1][1]) < 20:
            trade_profit = self.trades[-1][2]
            denominator = self.cash - trade_profit
            if denominator != 0 and (trade_profit / denominator) > 0.1:
                if action.action_type in ["open_long", "open_short"]:
                    reward -= 0.1

        self.current_step += 1
        done = self.current_step >= self.max_bars
        if done and self.position_shares != 0:
            final_price = self._get_observation().tf_1m.ohlcv.close
            if self.position_shares > 0:
                profit = self.position_shares * (final_price - self.entry_price)
            else:
                profit = -self.position_shares * (self.entry_price - final_price)
            reward += profit / 1000.0

        # *** FIX: initialise drawdown BEFORE using it ***
        drawdown = 0.0

        if done:
            if len(self.trades) > 1:
                returns = [p for _,_,p in self.trades]
                sharpe = np.mean(returns) / (np.std(returns) + 1e-9)
                reward += sharpe * 0.1
            final_value = self.cash + (self.position_shares * current_price if self.position_shares != 0 else 0)
            drawdown = (self.peak_value - final_value) / self.peak_value if self.peak_value > 0 else 0
            reward -= drawdown * 0.2

        total_profit = sum(p for _,_,p in self.trades)
        wins = sum(1 for _,_,p in self.trades if p>0)
        win_rate = wins / len(self.trades) if self.trades else 0.0
        info = Info(profit=total_profit, drawdown=drawdown, win_rate=win_rate, total_trades=len(self.trades))

        obs = self._get_observation()
        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def state(self) -> dict:
        return {
            "cash": self.cash,
            "position": self.position_shares,
            "step": self.current_step,
        }