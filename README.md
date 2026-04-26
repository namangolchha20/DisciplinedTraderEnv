# DisciplinedTraderEnv

## Problem
Retail traders lose money due to psychological biases, not lack of strategy. This environment trains an LLM agent to trade using technical analysis while enforcing strict risk management (max 30% account risk, trailing stop‑loss, no overtrading) and removing emotional feedback (hidden P&L).

## Environment
- Multi‑timeframe data: 1m, 5m, 15m, 1h, 1d with OHLCV, RSI, SMA20, Bollinger Bands, SuperTrend, Volume Index.
- Candlestick patterns (doji, hammer, engulfing, morning/evening star, etc.) and chart patterns (head & shoulders, double top/bottom, triangles, flags, cup & handle).
- Actions: open long/short, close position, set stop‑loss, do nothing.
- Reward components: profit/loss, overtrade penalty, patience bonus, trend alignment, stop‑loss compliance, pattern‑based bonus, risk‑adjusted (Sharpe, drawdown).
- Three difficulty levels (easy, medium, hard) with deterministic graders.

## Training
Uses GRPO + Unsloth on `unsloth/gemma-3-1b-it`. Curriculum learning automatically increases difficulty. Baseline comparisons (random, SMA crossover) show improvement.

## Results
Baseline random: ~0.2 reward; SMA crossover: ~0.4; trained agent after 500 rollouts: >0.7 reward and positive Sharpe ratio.

## How to run
1. `docker build -t disciplined_trader . && docker run -p 7860:7860 disciplined_trader`
2. Visit `http://localhost:7860/docs` to interact.
3. Training: see `train.ipynb` (Colab) or `inference.py`.

## Links
- Hugging Face Space: [your-space-url]
- Colab notebook: [link]
- Blog post / Video: [link]