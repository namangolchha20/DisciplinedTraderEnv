import uvicorn
import json
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openenv.core.env_server import create_app
from env.models import Action, Observation
from env.environment import DisciplinedTraderEnv

# Global variables
model = None
tokenizer = None
sessions = {}

app = create_app(DisciplinedTraderEnv, Action, Observation, env_name="disciplined_trader_env")

# Allow the frontend (port 5173) to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load_agent():
    global model, tokenizer
    try:
        from unsloth import FastLanguageModel
        print("Loading trained agent from ./trained_trader_lora...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            "./trained_trader_lora",
            max_seq_length=1024,
            dtype=torch.float16,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        print("Agent loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load trained agent. If you are training right now, this is expected: {e}")

class PredictRequest(BaseModel):
    observation: dict
    seed: int = 42
    step: int = 0

@app.post("/api/reset")
def api_reset():
    session_id = str(uuid.uuid4())
    env = DisciplinedTraderEnv()
    obs = env.reset(task_id="easy")
    sessions[session_id] = env
    return {"session_id": session_id, "observation": obs}

@app.post("/api/step/{session_id}")
def api_step(session_id: str, action: Action):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    env = sessions[session_id]
    result = env.step(action)
    
    # Fast-forward 5 bars so UI testers can quickly see market changes
    for _ in range(5):
        if result.done: break
        result = env.step(Action(action_type="do_nothing", amount_shares=0))
        
    return {"observation": result.observation, "reward": result.reward, "done": result.done, "info": result.info}

@app.post("/agent/predict")
def predict_action(req: PredictRequest):
    global model, tokenizer
    if model is None or tokenizer is None:
        import random
        # Fallback to random action to avoid 503 errors during UI testing
        action_type = random.choice(["open_long", "open_short", "close_position", "do_nothing"])
        return {"action_type": action_type, "amount_shares": 10}
        
    obs = req.observation
    prompt = (f"[SEED:{req.seed}][STEP:{req.step}]\n"
              f"Observation: cash={obs.get('cash', 0):.0f}, value={obs.get('account_value', 0):.0f}, "
              f"pos={obs.get('position_shares', 0)}, price={obs.get('tf_1m', {}).get('ohlcv', {}).get('close', 0):.2f}\n"
              f"Regime: {obs.get('market_regime', 'unknown')}, Pattern: {obs.get('tf_1m', {}).get('chart_pattern', 'unknown')}\n"
              "Valid action_types: 'open_long', 'open_short', 'close_position', 'do_nothing'\n"
              "Generate an action in JSON: {\"action_type\": \"...\", \"amount_shares\": 0}")
    
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=64, pad_token_id=tokenizer.pad_token_id)
    completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    
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


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()