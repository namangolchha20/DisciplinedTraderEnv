# 📉 The ₹2.5 Lakh Lesson: Teaching LLMs the Discipline I Didn't Have

**By Naman Golchha** | *A submission for the OpenEnv Hackathon — Theme #3: World Modeling*

---

## The "Josh" and the Fall

I started trading in January 2022 when I was in 10th grade. At first, it felt like magic. Fueled by beginner's luck and reading finance books, I made roughly ₹4 Lakhs in just 15 days. I bought my own laptop, tablet, and smartphone with my own money. I was convinced I was going to get rich quick and retire early.

My parents warned me to take half the money out and put it in a bank account. But I was in the **"josh"** (frenzy) of making more money. My ego took over.

Then came the reality check. Just days before my Class 10 Science board exam, I hit the biggest loss of my life: **₹2.5 Lakhs**. Why? Because I refused to accept I was wrong. I held a losing options trade for two days, praying it would go to breakeven, and watched helplessly as theta decay absolutely crushed my account. In that moment, I wasn't trading anymore; I was purely gambling.

I stopped trading shortly after that as soon as I reached class 11 to focus on my JEE preparation.

But I always wondered: *What if I had just closed that trade early when it hit my stop loss?* Humans are psychologically connected to their money. We freeze. We hope. We gamble.

**But AI agents? They just follow instructions.**

---

## The Problem: LLMs are Amateur Traders

When we build Large Language Model trading agents today, they act exactly like amateur human traders. Because they treat trading simply as a "next-token prediction" task, they revenge-trade, hallucinate invalid actions, over-leverage, and blow up accounts. They lack discipline.

To solve this, I built **DisciplinedTraderEnv**.

While most trading environments focus purely on predicting price movements, DisciplinedTraderEnv is different. It is built to mathematically enforce the 9 hard-learned lessons of an experienced trader.

---

## 🌍 The Solution: DisciplinedTraderEnv

Built on the newest release of the OpenEnv framework, DisciplinedTraderEnv is a partially observable, multi‑timeframe financial simulator.

Instead of just rewarding the agent for "making money," I built a complex reward system designed to enforce discipline:

- **The Overtrading Penalty:** Every time the agent takes a trade, its score is reduced. This teaches the agent to sit on its hands and wait for high‑probability setups, rather than constantly pressing buttons.
- **The Verifier:** A strict format verifier ensures the LLM outputs perfect JSON schemas (`open_long`, `close_position`, etc.). Invalid actions are heavily punished.
- **Strategic Alignment:** The agent is rewarded for patience—waiting for valid candlestick patterns and aligning with the broader market regime, while being penalized for noisy trading.

---

## 🧠 The Training Pipeline (GRPO & Unsloth)

To teach the agent these rules, I used Reinforcement Learning. Using **Qwen2.5-1.5B-Instruct**, I trained the agent via **GRPO (Generative Reward Policy Optimization)**. To make this efficient enough to run on a single T4 GPU, the training loop was heavily optimized using **Unsloth** and **HF TRL**.

The agent must pass multiple independent verifiers. It cannot simply hold a position forever because time penalties exist; it cannot spam trades because transaction costs destroy its PnL; and it cannot cheat because it is sandbox‑isolated from the global state.

---

## 🏆 The Results

The results of the GRPO training proved that discipline can be learned. As seen in the training logs, the agent underwent two distinct learning phases:

- **Phase 1 (Risk Aversion):** The agent rapidly learned that random trading destroys capital. It stabilized its losses by learning *not* to trade bad setups. It learned the hardest lesson of all: sometimes the best trade is no trade.
- **Phase 2 (Format Mastery):** The agent perfectly mastered the JSON formatting constraint, pushing the formatting verifier reward to its maximum.

When evaluated against an untrained **Random** baseline (which rapidly blew up the account, much like a frantic human trader), our trained agent survived—preserving capital, adhering strictly to the risk management rules, and executing flawless JSON workflows.

I know this environment isn't perfect yet, and I plan to keep improving it in the future. But by combining OpenEnv, Unsloth, and GRPO, I was able to give an LLM the one thing I lacked as a 10th grader: **cold, mathematical discipline.**
