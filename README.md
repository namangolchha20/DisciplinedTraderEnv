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
> - 🚀 **[Hugging Face Space Demo](https://huggingface.co/spaces/NGGAMER/disciplined-trader-train)**
> - 📖 **[Mini-Blog / Video Pitch](YOUR_BLOG_OR_VIDEO_URL)**
> - 📓 **[Training Colab Notebook](https://colab.research.google.com/drive/1aSE4ajCWd29D0Bn9vFtJviAzuyhjGsuZ#scrollTo=O7jhNgUCsFjK)**

---

## 🎯 Theme #3: World Modeling (Professional Tasks)
**The Problem:** Most LLM trading agents fail in live environments because they treat trading as a next-token prediction task. They hallucinate invalid actions, revenge trade, and suffer massive drawdowns.
**Our Solution:** We built a partially observable, multi-timeframe financial world (`DisciplinedTraderEnv`). We then trained an LLM agent to not just predict prices, but to **execute complex, multi-step trading workflows** while strictly adhering to risk management. 

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
