---
title: Disciplined Trader LLM Agent
emoji: 📈
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---
# DisciplinedTraderEnv: LLM Trading Agent 🤖📈

An advanced RL environment and training pipeline that teaches a Large Language Model (LLM) to trade profitably using technical analysis and strict risk management.

## 🎯 The Problem
Retail traders often fail due to psychological biases (fear, greed, revenge trading) rather than a lack of strategy. This environment is designed to train an LLM agent to make objective trading decisions based purely on multi-timeframe technical indicators and chart patterns, enforcing:
- Maximum 30% account risk.
- Trailing stop-losses.
- Strict penalties for overtrading.
- Hidden P&L during the trade to prevent emotional feedback loops.

## 🧠 The Agent & Reinforcement Learning (GRPO)
Instead of relying on hardcoded heuristics, we train the LLM using **GRPO (Generative Reward Policy Optimization)**. 
- **Model**: `unsloth/Qwen2.5-1.5B-Instruct` loaded in 4-bit quantization.
- **LoRA**: Parameter-Efficient Fine-Tuning (PEFT) adapters are attached to make training possible on a single T4 GPU.
- **Reward Shaping**: The agent is heavily rewarded for adhering to a strict JSON action format, and then iteratively optimizes for maximum environment profit, patience bonuses, and trend alignment. 

## 📊 Environment Features
- **Multi-timeframe data**: 1m, 5m, 15m, 1h, 1d (OHLCV, RSI, SMA20, Bollinger Bands, SuperTrend).
- **Candlestick & Chart Patterns**: Detects engulfing, doji, head & shoulders, triangles, etc.
- **Actions**: The LLM outputs strict JSON to `open_long`, `open_short`, `close_position`, or `do_nothing`.
- **Three difficulty levels**: Easy, medium, and hard with curriculum learning.

## 🚀 How to Run

### 1. Training the Agent
To kick off the GRPO training pipeline locally or in Google Colab:
```bash
python inference.py
```
This will train the LoRA adapters over 3 epochs and save them to `./trained_trader_lora`.

### 2. Evaluating the Agent
To test the trained LLM agent against unseen historical data and compare it to an SMA crossover baseline:
```bash
python evaluate.py
```

### 3. Running the FastAPI Backend
Start the server to expose the environment and the trained LLM prediction endpoint (`/agent/predict`):
```bash
docker build -t disciplined_trader . 
docker run -p 7860:7860 disciplined_trader
```
Or run directly:
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```
Visit `http://localhost:7860/docs` to interact with the API!

## 📈 Results
- **Random Baseline**: ~(-19.0) Reward
- **SMA Crossover Baseline**: ~(-0.5) Reward
- **Trained LLM Agent**: Learns to output perfect JSON schema and achieves positive profitability by aligning with the environment's strict risk-management heuristics.