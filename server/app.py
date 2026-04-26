import uvicorn
import json
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openenv.core.env_server import create_app
from env.models import Action, Observation
from env.environment import DisciplinedTraderEnv

# Global model variables
model = None
tokenizer = None

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
    observation: Observation
    seed: int = 42
    step: int = 0

@app.post("/agent/predict")
def predict_action(req: PredictRequest):
    global model, tokenizer
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    obs = req.observation
    prompt = (f"[SEED:{req.seed}][STEP:{req.step}]\n"
              f"Observation: cash={obs.cash:.0f}, value={obs.account_value:.0f}, "
              f"pos={obs.position_shares}, price={obs.tf_1m.ohlcv.close:.2f}\n"
              f"Regime: {obs.market_regime}, Pattern: {obs.tf_1m.chart_pattern}\n"
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