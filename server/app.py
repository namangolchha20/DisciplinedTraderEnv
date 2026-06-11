import json
import os
import sys
import threading
from pathlib import Path
from typing import Optional, Dict, List

import uvicorn
from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from openenv.core.env_server import create_app
from env.models import Action, Observation
from env.environment import DisciplinedTraderEnv
from env.graders import grade
from env.policies import POLICIES, confluence_score, CONFLUENCE_MAX, ENTRY_SCORE, ENTRY_MARGIN, llm_position_overlay
from env.model_loader import load_trained_agent, ensure_adapter_downloaded

# ----------------------------------------------------------------------
# OpenEnv-compliant app (exposes /reset, /step, /state for RL clients)
# ----------------------------------------------------------------------
app = create_app(DisciplinedTraderEnv, Action, Observation, env_name="disciplined_trader_env")

STATIC_DIR = Path(__file__).resolve().parent / "static"

# ----------------------------------------------------------------------
# Optional trained-LLM agent (loaded if a LoRA checkpoint exists).
# The adapter is HOT-RELOADED whenever inference.py saves a newer version,
# so train -> watch -> train cycles never require a server restart.
# ----------------------------------------------------------------------
model = None
tokenizer = None
_adapter_loaded_stamp = None
_agent_lock = threading.Lock()
_load_error: Optional[str] = None
ADAPTER_DIR = Path("./trained_trader_lora")
HF_ADAPTER_REPO = os.environ.get("HF_ADAPTER_REPO", "NGGAMER/disciplined-trader-lora")


def _ensure_adapter_files() -> bool:
    """Local adapter or download from Hugging Face Hub."""
    if (ADAPTER_DIR / "adapter_config.json").exists():
        return True
    try:
        ensure_adapter_downloaded(ADAPTER_DIR, HF_ADAPTER_REPO)
        return (ADAPTER_DIR / "adapter_config.json").exists()
    except Exception as e:
        global _load_error
        err = str(e)
        if "404" in err or "Repository Not Found" in err:
            _load_error = (
                f"Model repo not found: huggingface.co/{HF_ADAPTER_REPO}. "
                "Upload adapter files to that Model repo first."
            )
        elif "401" in err or "403" in err:
            _load_error = f"Cannot access {HF_ADAPTER_REPO} — make the model repo public."
        else:
            _load_error = f"Could not download adapter from {HF_ADAPTER_REPO}: {err}"
        print(f"Warning: {_load_error}")
        return False


def _adapter_stamp():
    """Modification time of a local adapter, or None if not present."""
    try:
        return (ADAPTER_DIR / "adapter_config.json").stat().st_mtime
    except OSError:
        return None


def _adapter_available() -> bool:
    """True if LLM mode can be attempted (local files or Hub repo configured)."""
    if (ADAPTER_DIR / "adapter_config.json").exists():
        return True
    return bool(HF_ADAPTER_REPO)


def load_agent():
    global model, tokenizer, _adapter_loaded_stamp, _load_error
    if not _ensure_adapter_files():
        return
    stamp = _adapter_stamp()
    if stamp is None:
        return
    try:
        import torch
        if not torch.cuda.is_available():
            _load_error = (
                "LLM mode needs a GPU. On Hugging Face Spaces, set Hardware to a GPU tier "
                "(e.g. T4 small). Bot/SMA/Random work on CPU."
            )
            return
        print(f"Loading trained agent from {ADAPTER_DIR} (python: {sys.executable})...")
        model, tokenizer = load_trained_agent(ADAPTER_DIR)
        _adapter_loaded_stamp = stamp
        _load_error = None
        print("Agent loaded successfully!")
    except Exception as e:
        _load_error = str(e)
        if "bitsandbytes" in _load_error.lower() or "cuda" in _load_error.lower():
            _load_error = (
                "LLM failed to load (GPU/CUDA required). Use Disciplined Bot on CPU, "
                f"or enable GPU hardware on the Space. Detail: {e}"
            )
        print(f"Warning: Could not load trained agent: {_load_error}")


def maybe_reload_agent():
    """Reload the adapter if inference.py saved a newer one since last load."""
    stamp = _adapter_stamp()
    if stamp is not None and stamp != _adapter_loaded_stamp:
        load_agent()


def ensure_agent_loaded() -> bool:
    """Load the LLM on first use so the web terminal starts without GPU deps."""
    with _agent_lock:
        maybe_reload_agent()
        if model is None and _adapter_available():
            load_agent()
    return model is not None and tokenizer is not None


def _llm_unavailable_detail() -> str:
    if not (ADAPTER_DIR / "adapter_config.json").exists() and not _ensure_adapter_files():
        return (
            f"No trained adapter found. Train locally (`inference.py`) or upload to "
            f"huggingface.co/{HF_ADAPTER_REPO} and set HF_ADAPTER_REPO if needed."
        )
    if _load_error and "No module named 'torch'" in _load_error:
        return (
            "PyTorch is not installed in this Python. "
            "Start the server with .\\run_server.ps1"
        )
    if _load_error:
        return f"Trained adapter found but failed to load: {_load_error}"
    return "Trained LLM is not loaded yet."


class PredictRequest(BaseModel):
    observation: Observation
    seed: int = 42
    step: int = 0
    task: str = "easy"


def _build_prompt(obs: Observation, seed: int, step: int, task: str = "easy") -> str:
    # Must mirror the training prompt template in inference.py exactly.
    return (f"[SEED:{seed}][STEP:{step}][TASK:{task}]\n"
            f"Observation: cash={obs.cash:.0f}, value={obs.account_value:.0f}, "
            f"pos={obs.position_shares}, price={obs.tf_1m.ohlcv.close:.2f}\n"
            f"Regime: {obs.market_regime}, Pattern: {obs.tf_1m.chart_pattern}\n"
            "Valid action_types: 'open_long', 'open_short', 'close_position', 'do_nothing'\n"
            "Generate an action in JSON: {\"action_type\": \"...\", \"amount_shares\": 0}")


def _parse_action(completion: str) -> Action:
    import re
    try:
        json_match = re.search(r'\{.*\}', completion, re.DOTALL)
        if json_match:
            action_dict = json.loads(json_match.group())
            return Action(
                action_type=action_dict.get("action_type", "do_nothing"),
                amount_shares=action_dict.get("amount_shares", 0)
            )
    except Exception:
        pass
    return Action(action_type="do_nothing", amount_shares=0)


def _llm_action(obs: Observation, seed: int, step: int, task: str = "easy") -> Action:
    import torch
    prompt = _build_prompt(obs, seed, step, task)
    device = next(model.parameters()).device
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return _parse_action(completion)


@app.post("/agent/predict")
def predict_action(req: PredictRequest):
    if not ensure_agent_loaded():
        raise HTTPException(status_code=503, detail=_llm_unavailable_detail())
    return _llm_action(req.observation, req.seed, req.step, req.task)


# ----------------------------------------------------------------------
# Interactive demo session powering the web trading terminal
# ----------------------------------------------------------------------
TF_FACTOR = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "1d": 390}
TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "1d": 86400}
BASE_TS = 1736065800  # Mon 2025-01-05 09:30 UTC — purely cosmetic chart axis


class DemoSession:
    def __init__(self):
        self.lock = threading.Lock()
        self.env: Optional[DisciplinedTraderEnv] = None
        self.task = "easy"
        self.seed = 42
        self.reward_total = 0.0
        self.done = False
        self.sent: Dict[str, int] = {}
        self.equity_sent = 0
        self.trades_sent = 0


SESSION = DemoSession()


class ResetRequest(BaseModel):
    task: str = "easy"
    seed: int = 42


class StepRequest(BaseModel):
    mode: str = "manual"            # manual | bot | random | sma
    steps: int = 1
    action_type: str = "do_nothing"
    amount_shares: int = 0
    stop_loss_percent: Optional[float] = None


def _tf_series(env: DisciplinedTraderEnv, tf: str):
    return {
        "1m": env.ohlcv_1m, "5m": env.ohlcv_5m, "15m": env.ohlcv_15m,
        "1h": env.ohlcv_1h, "1d": env.ohlcv_1d,
    }[tf]


def _candle_dict(bar, tf: str, idx: int):
    return {
        "time": BASE_TS + idx * TF_SECONDS[tf],
        "open": round(bar.open, 4), "high": round(bar.high, 4),
        "low": round(bar.low, 4), "close": round(bar.close, 4),
        "volume": round(bar.volume, 2),
    }


def _new_candles(sess: DemoSession):
    """Candles revealed since the last payload, per timeframe."""
    env = sess.env
    out = {}
    cur = min(env.current_step, len(env.ohlcv_1m) - 1)
    for tf, factor in TF_FACTOR.items():
        series = _tf_series(env, tf)
        visible = min(cur // factor + 1, len(series))
        start = sess.sent.get(tf, 0)
        out[tf] = [_candle_dict(series[i], tf, i) for i in range(start, visible)]
        sess.sent[tf] = visible
    return out


def _tf_summary(t):
    return {
        "rsi": round(t.rsi, 2),
        "sma": round(t.moving_average, 4),
        "bb_upper": round(t.bb_upper, 4),
        "bb_lower": round(t.bb_lower, 4),
        "supertrend": t.super_trend_direction,
        "volume_index": round(t.volume_index, 2),
        "support": round(t.support, 4),
        "resistance": round(t.resistance, 4),
        "candle": t.candlestick_pattern,
        "chart": t.chart_pattern,
    }


def _obs_summary(obs: Observation):
    bull, bear, breakdown = confluence_score(obs)
    return {
        "regime": obs.market_regime,
        "risk_usage": round(obs.risk_usage, 4),
        "time_since_last_trade": obs.time_since_last_trade,
        "confluence": {
            "bull": bull, "bear": bear, "max": round(CONFLUENCE_MAX, 1),
            "entry_score": ENTRY_SCORE, "entry_margin": ENTRY_MARGIN,
            "breakdown": breakdown,
        },
        "tf": {
            "1m": _tf_summary(obs.tf_1m), "5m": _tf_summary(obs.tf_5m),
            "15m": _tf_summary(obs.tf_15m), "1h": _tf_summary(obs.tf_1h),
            "1d": _tf_summary(obs.tf_1d),
        },
    }


def _account(sess: DemoSession):
    env = sess.env
    price = env._current_price()
    equity = env.account_value()
    unrealized = env.position_shares * (price - env.entry_price) if env.position_shares else 0.0
    wins = sum(1 for _, _, p in env.trades if p > 0)
    realized = sum(p for _, _, p in env.trades)
    return {
        "cash": round(env.cash, 2),
        "equity": round(equity, 2),
        "price": round(price, 4),
        "position_shares": env.position_shares,
        "entry_price": round(env.entry_price, 4),
        "stop_loss": round(env.stop_loss, 4) if env.stop_loss else None,
        "unrealized": round(unrealized, 2),
        "realized": round(realized, 2),
        "peak": round(env.peak_value, 2),
        "max_drawdown": round(env.max_drawdown, 4),
        "win_rate": round(wins / len(env.trades), 4) if env.trades else 0.0,
        "total_trades": len(env.trades),
        "bar": env.current_step,
        "max_bars": env.max_bars,
    }


def _new_equity(sess: DemoSession):
    env = sess.env
    pts = [
        {"time": BASE_TS + i * 60, "value": round(v, 2)}
        for i, v in enumerate(env.equity_curve)
    ][sess.equity_sent:]
    sess.equity_sent = len(env.equity_curve)
    return pts


def _new_trades(sess: DemoSession):
    env = sess.env
    out = [
        {"entry_bar": e, "exit_bar": x, "profit": round(p, 2)}
        for e, x, p in env.trades[sess.trades_sent:]
    ]
    sess.trades_sent = len(env.trades)
    return out


@app.get("/api/status")
def demo_status():
    loaded = model is not None and tokenizer is not None
    return {
        "llm_available": _adapter_available(),
        "llm_loaded": loaded,
        "load_error": _load_error,
        "python": sys.executable,
    }


@app.post("/api/reset")
def demo_reset(req: ResetRequest):
    with SESSION.lock:
        SESSION.env = DisciplinedTraderEnv()
        SESSION.task = req.task if req.task in ("easy", "medium", "hard") else "easy"
        SESSION.seed = req.seed
        SESSION.reward_total = 0.0
        SESSION.done = False
        SESSION.sent = {}
        SESSION.equity_sent = 0
        SESSION.trades_sent = 0
        obs = SESSION.env.reset(task_id=SESSION.task, seed=SESSION.seed)
        return {
            "task": SESSION.task,
            "seed": SESSION.seed,
            "candles": _new_candles(SESSION),
            "equity": _new_equity(SESSION),
            "account": _account(SESSION),
            "obs": _obs_summary(obs),
            "reward_total": 0.0,
            "done": False,
        }


@app.post("/api/step")
def demo_step(req: StepRequest):
    with SESSION.lock:
        if SESSION.env is None:
            raise HTTPException(status_code=400, detail="Call /api/reset first.")
        if SESSION.done:
            raise HTTPException(status_code=400, detail="Episode finished. Reset to start a new one.")

        env = SESSION.env
        steps = max(1, min(req.steps, 500))
        if req.mode == "llm":
            if not ensure_agent_loaded():
                raise HTTPException(status_code=400, detail=_llm_unavailable_detail())
            steps = 1  # one GPU inference per request (~6s on 4GB GPUs)
        policy = POLICIES.get(req.mode)
        markers: List[dict] = []
        reward_step = 0.0
        last_info = None

        for i in range(steps):
            if req.mode == "manual":
                # The chosen action fires on the first bar; remaining bars hold.
                if i == 0:
                    action = Action(
                        action_type=req.action_type,
                        amount_shares=req.amount_shares,
                        stop_loss_percent=req.stop_loss_percent,
                    )
                else:
                    action = Action(action_type="do_nothing", amount_shares=0)
            elif req.mode == "llm":
                llm_action = _llm_action(env._cached_obs, SESSION.seed, env.current_step, SESSION.task)
                action = llm_position_overlay(env._cached_obs, llm_action)
            elif policy is not None:
                action = policy(env._cached_obs)
            else:
                raise HTTPException(status_code=400, detail=f"Unknown mode '{req.mode}'.")

            bar = env.current_step
            pos_before = env.position_shares
            price = env._current_price()
            result = env.step(action)
            reward_step += result.reward
            last_info = result.info

            # Detect actual fills (covers stop-loss exits and rejections too).
            if pos_before == 0 and env.position_shares != 0:
                markers.append({
                    "bar": bar, "price": round(price, 4),
                    "type": "long" if env.position_shares > 0 else "short",
                    "shares": abs(env.position_shares),
                })
            elif pos_before != 0 and env.position_shares == 0:
                profit = env.trades[-1][2] if env.trades else 0.0
                kind = "close" if action.action_type == "close_position" else "stop"
                markers.append({
                    "bar": bar, "price": round(price, 4),
                    "type": kind, "profit": round(profit, 2),
                })

            if result.done:
                SESSION.done = True
                break

        SESSION.reward_total += reward_step
        payload = {
            "candles": _new_candles(SESSION),
            "equity": _new_equity(SESSION),
            "trades": _new_trades(SESSION),
            "markers": markers,
            "account": _account(SESSION),
            "obs": _obs_summary(env._cached_obs),
            "reward_step": round(reward_step, 4),
            "reward_total": round(SESSION.reward_total, 4),
            "done": SESSION.done,
            "grade": round(grade(env, SESSION.task), 4) if SESSION.done else None,
            "info": last_info.model_dump() if last_info else None,
        }
        return payload


# ----------------------------------------------------------------------
# Frontend
# ----------------------------------------------------------------------
@app.get("/ui", include_in_schema=False)
def ui():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
