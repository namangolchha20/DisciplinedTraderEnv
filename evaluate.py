import os
import json
import torch
import numpy as np
from env.environment import DisciplinedTraderEnv
from env.models import Action

# ------------------------------------------------------------
# Baseline Policy
# ------------------------------------------------------------
def sma_crossover_policy(obs, env):
    if hasattr(obs, 'tf_1h') and hasattr(obs.tf_1h, 'moving_average') and hasattr(obs.tf_1h, 'sma50'):
        if obs.tf_1h.moving_average > obs.tf_1h.sma50:
            if env.position_shares == 0:
                return Action(action_type="open_long", amount_shares=20)
            elif env.position_shares < 0:
                return Action(action_type="close_position", amount_shares=0)
        elif obs.tf_1h.moving_average < obs.tf_1h.sma50:
            if env.position_shares == 0:
                return Action(action_type="open_short", amount_shares=20)
            elif env.position_shares > 0:
                return Action(action_type="close_position", amount_shares=0)
    return Action(action_type="do_nothing", amount_shares=0)

# ------------------------------------------------------------
# Agent Policy wrapper
# ------------------------------------------------------------
class LLMTradingAgent:
    def __init__(self, model_path="./trained_trader_lora"):
        from unsloth import FastLanguageModel
        print(f"Loading trained agent from {model_path}...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_path,
            max_seq_length=1024,
            dtype=torch.float16,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)

    def get_action(self, obs, env, seed_val, step_val):
        prompt = (f"[SEED:{seed_val}][STEP:{step_val}]\n"
                  f"Observation: cash={obs.cash:.0f}, value={obs.account_value:.0f}, "
                  f"pos={obs.position_shares}, price={obs.tf_1m.ohlcv.close:.2f}\n"
                  f"Regime: {obs.market_regime}, Pattern: {obs.tf_1m.chart_pattern}\n"
                  "Valid action_types: 'open_long', 'open_short', 'close_position', 'do_nothing'\n"
                  "Generate an action in JSON: {\"action_type\": \"...\", \"amount_shares\": 0}")
        
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=64, pad_token_id=self.tokenizer.pad_token_id)
        
        completion = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        
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

def evaluate(policy_func, name="Policy", num_episodes=5, task="easy"):
    env = DisciplinedTraderEnv()
    rewards = []
    account_values = []
    
    print(f"\nEvaluating {name} over {num_episodes} episodes...")
    for ep in range(num_episodes):
        seed_val = 100 + ep # Unseen seed for eval
        obs = env.reset(task_id=task, seed=seed_val)
        total = 0.0
        done = False
        step = 0
        
        while not done:
            if isinstance(policy_func, LLMTradingAgent):
                action = policy_func.get_action(obs, env, seed_val, step)
            else:
                action = policy_func(obs, env)
                
            result = env.step(action)
            total += result.reward
            done = result.done
            obs = result.observation
            step += 1
            
        rewards.append(total)
        account_values.append(obs.account_value)
        print(f"Episode {ep+1}: Reward = {total:.2f}, Final Account Value = ${obs.account_value:.2f}")
        
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_acc = np.mean(account_values)
    
    print(f"\n--- {name} Results ---")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Account Value: ${mean_acc:.2f}")
    return mean_reward, mean_acc

if __name__ == "__main__":
    try:
        agent = LLMTradingAgent()
        evaluate(agent, name="Trained LLM Agent", num_episodes=5)
    except Exception as e:
        print(f"Could not load LLM agent (Make sure ./trained_trader_lora exists!): {e}")
        
    evaluate(sma_crossover_policy, name="SMA Crossover Baseline", num_episodes=5)
