import os
import re
import json
import numpy as np
import torch
from env.environment import DisciplinedTraderEnv
from env.models import Action
from env.policies import sma_crossover_policy, disciplined_bot, random_policy, llm_position_overlay
from env.graders import grade
from env.model_loader import load_trained_agent

# ------------------------------------------------------------
# Agent Policy wrapper
# ------------------------------------------------------------
class LLMTradingAgent:
    def __init__(self, model_path="./trained_trader_lora"):
        print(f"Loading trained agent from {model_path}...")
        self.model, self.tokenizer = load_trained_agent(model_path)

    def get_action(self, obs, env, seed_val, step_val):
        # Must mirror the training prompt template in inference.py exactly.
        prompt = (f"[SEED:{seed_val}][STEP:{step_val}][TASK:{env.task}]\n"
                  f"Observation: cash={obs.cash:.0f}, value={obs.account_value:.0f}, "
                  f"pos={obs.position_shares}, price={obs.tf_1m.ohlcv.close:.2f}\n"
                  f"Regime: {obs.market_regime}, Pattern: {obs.tf_1m.chart_pattern}\n"
                  "Valid action_types: 'open_long', 'open_short', 'close_position', 'do_nothing'\n"
                  "Generate an action in JSON: {\"action_type\": \"...\", \"amount_shares\": 0}")

        device = next(self.model.parameters()).device
        inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        completion = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )

        try:
            json_match = re.search(r'\{.*\}', completion, re.DOTALL)
            if json_match:
                action_dict = json.loads(json_match.group())
                llm_action = Action(
                    action_type=action_dict.get("action_type", "do_nothing"),
                    amount_shares=action_dict.get("amount_shares", 0),
                )
            else:
                llm_action = Action(action_type="do_nothing", amount_shares=0)
        except Exception:
            llm_action = Action(action_type="do_nothing", amount_shares=0)

        return llm_position_overlay(obs, llm_action)

def evaluate(policy_func, name="Policy", num_episodes=5, task="easy"):
    env = DisciplinedTraderEnv()
    rewards = []
    account_values = []
    grades = []
    
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
                action = policy_func(obs)
                
            result = env.step(action)
            total += result.reward
            done = result.done
            obs = result.observation
            step += 1
            
        rewards.append(total)
        account_values.append(obs.account_value)
        grades.append(grade(env, task))
        print(f"Episode {ep+1}: Reward = {total:.2f}, Final Account Value = ${obs.account_value:.2f}, Grade = {grades[-1]:.3f}")
        
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_acc = np.mean(account_values)
    mean_grade = np.mean(grades)
    
    print(f"\n--- {name} Results ---")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Account Value: ${mean_acc:.2f}")
    print(f"Mean Grade: {mean_grade:.3f}")
    return mean_reward, mean_acc

if __name__ == "__main__":
    try:
        agent = LLMTradingAgent()
        evaluate(agent, name="Trained LLM Agent", num_episodes=5)
    except Exception as e:
        print(f"Could not load LLM agent (Make sure ./trained_trader_lora exists!): {e}")
        
    evaluate(sma_crossover_policy, name="SMA Crossover Baseline", num_episodes=5)
    evaluate(disciplined_bot, name="Disciplined Rule Bot", num_episodes=5)
    evaluate(random_policy, name="Random Policy", num_episodes=5)
