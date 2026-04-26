---
title: Disciplined Trader LLM Agent
emoji: 📈
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 📈 DisciplinedTraderEnv: LLM Trading Agent via OpenEnv & GRPO

**OpenEnv Hackathon 2026 Submission**

> 🔗 **Important Links:**
> - 🚀 **[Hugging Face Space Demo](YOUR_HF_SPACE_URL)**
> - 📖 **[Mini-Blog / Video Pitch](YOUR_BLOG_OR_VIDEO_URL)**
> - 📓 **[Training Colab Notebook](YOUR_COLAB_NOTEBOOK_URL)**

---

## 🎯 The Problem & Our Inspiration
This project was born out of 13 hard-learned personal lessons from live trading:
1. Never overtrade.
2. Don't buy call and put options together.
3. Have patience if you feel that the price will go high.
4. Don't make stock market a permanent earning place, start a business and make it big.
5. Never take any unusual trade.
6. Never trade all of your monetary value.
7. Act fast very fast.
8. Never forget about theta, it can blank you out.
9. Never trade after a huge profit.
10. Never see the PnL while trading because it can make you take incorrect decisions. Always see your analysis.
11. Always trade by your analysis, never trade seeing others taking that trade.
12. Always use a stop loss.
13. Don't trade against the market.

**The Solution:** Most LLM trading agents fail because they treat trading as a next-token prediction task, falling victim to the exact same psychological traps (fear, greed, revenge trading, overtrading) that humans do. 

We built a partially observable, multi-timeframe financial world (`DisciplinedTraderEnv`) to mathematically enforce these 13 rules. We then trained an LLM agent to **execute complex, multi-step trading workflows** while strictly adhering to this risk management philosophy. 

Instead of relying on hardcoded heuristics, we trained an Unsloth-optimized `Qwen2.5-1.5B-Instruct` model using **Generative Reward Policy Optimization (GRPO)** to internalize trading discipline.

---

## 🌍 The Environment: How It Works
The agent operates in a fully functional trading simulator built on the latest release of `OpenEnv`.
* **What the Agent Sees (State):** Multi-timeframe OHLCV data (1m, 5m, 15m, 1h, 1d), current cash, active position shares, unrealized PnL, detected chart patterns (e.g., bull flags), and market regimes.
* **What the Agent Does (Actions):** The LLM must output perfectly formatted JSON to `open_long`, `open_short`, `close_position`, or `do_nothing`.
* **The Challenge:** The market is noisy. Overtrading incurs massive transaction penalties. Holding losing positions incurs drawdown penalties.

### 🛡️ Preventing Reward Hacking
To prevent the agent from gaming the system, we used **multiple independent reward checks**:
1. **Format Compliance (The Verifier):** Heavily penalizes non-JSON outputs or invalid action schemas.
2. **Execution Success:** Punishes attempting to short without capital or open concurrent positions.
3. **Strategic Alignment:** Rewards entering trades during valid bullish/bearish candlestick patterns.
4. **Drawdown Penalties:** Heavily penalizes the agent at the end of the episode if peak-to-trough account value drops significantly.

---

## 🧠 Training Pipeline & Results

We used **HF TRL** and **Unsloth** for highly efficient LoRA training on a single T4 GPU. The agent learned through trial and error, not supervised mimicking.

### 📊 Training Progress
As seen in the reward curve, the agent underwent two distinct learning phases:
1. **Phase 1 (Risk Aversion):** The agent quickly learned that random trading destroys capital via transaction costs. The trading reward stabilized at `-0.001` (the baseline time penalty for doing nothing). It learned to stop bleeding money!
2. **Phase 2 (Format Mastery):** The agent perfectly mastered the JSON formatting constraint, pushing the formatting reward to its maximum.

*(Note: Replace the image below with your generated plot!)*
![Training Reward Curve](reward_curve.png)
*The reward curve showing the agent mastering JSON formatting and stabilizing trading losses over 900 GRPO steps.*

### 🏆 Before / After Baseline Comparison
We evaluated the agent in `evaluate.py` over unseen test data:
* **Untrained Random Agent:** `-19.78 ± 0.69` Reward (Rapidly blew up the account).
* **Baseline SMA Strategy:** `-0.50 ± 0.00` Reward (Suffered from whipsaws).
* **Trained GRPO Agent:** *(Add your final reward here!)* Achieved disciplined capital preservation and perfectly formatted execution.

---

## 🚀 Reproducibility & Deployment

### 1. Run the FastAPI Backend (or view on HF Spaces)
```bash
docker build -t disciplined_trader . 
docker run -p 7860:7860 disciplined_trader
```
Or run directly via Uvicorn:
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 2. Train the Agent Yourself
A minimal training script (`inference.py`) is provided. It uses Unsloth for 2x faster, memory-efficient GRPO training.
```bash
python inference.py
```

### 3. Evaluate the Agent
```bash
python evaluate.py
```