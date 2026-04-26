import os
import torch
import json
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from env.environment import DisciplinedTraderEnv
from env.models import Action

# Force GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ------------------------------------------------------------
# Baseline Policies (unchanged)
# ------------------------------------------------------------
def random_policy(obs, env=None):
    import random
    actions = ["open_long", "open_short", "close_position", "do_nothing"]
    chosen = random.choice(actions)
    if chosen in ["open_long", "open_short"]:
        return Action(action_type=chosen, amount_shares=random.randint(5, 20))
    else:
        return Action(action_type=chosen, amount_shares=0)

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

def evaluate_policy(policy_func, num_episodes=5, task="easy"):
    env = DisciplinedTraderEnv()
    rewards = []
    for ep in range(num_episodes):
        obs = env.reset(task_id=task, seed=42 + ep)
        total = 0.0
        done = False
        while not done:
            action = policy_func(obs, env) if callable(policy_func) else policy_func
            result = env.step(action)
            total += result.reward
            done = result.done
            obs = result.observation
        rewards.append(total)
    return np.mean(rewards), np.std(rewards)

# ------------------------------------------------------------
# Reward Function for GRPOTrainer (full episode)
# ------------------------------------------------------------
def reward_func(prompts, completions, **kwargs):
    """
    For each completion, run a full episode of the trading environment,
    using the completion as the policy to generate actions at each step.
    Returns a list of total episode rewards.
    """
    import re
    rewards = []
    for i in range(len(completions)):
        prompt_text = prompts[i] if isinstance(prompts[i], str) else str(prompts[i])
        comp_text = completions[i] if isinstance(completions[i], str) else str(completions[i])
        
        seed_match = re.search(r'\[SEED:(\d+)\]', prompt_text)
        step_match = re.search(r'\[STEP:(\d+)\]', prompt_text)
        
        total_reward = -0.1
        
        if seed_match and step_match:
            env_seed = int(seed_match.group(1))
            env_step = int(step_match.group(1))
            
            env = DisciplinedTraderEnv()
            obs = env.reset(task_id="easy", seed=env_seed)
            
            # Fast-forward to the exact state
            for _ in range(env_step):
                obs = env.step(Action("do_nothing", 0)).observation
            
            try:
                json_match = re.search(r'\{.*\}', comp_text, re.DOTALL)
                if json_match:
                    action_dict = json.loads(json_match.group())
                    action = Action(
                        action_type=action_dict.get("action_type", "do_nothing"),
                        amount_shares=action_dict.get("amount_shares", 0)
                    )
                else:
                    action = Action(action_type="do_nothing", amount_shares=0)
            except Exception:
                action = Action(action_type="do_nothing", amount_shares=0)
            
            # Execute ONE single action in the environment and get immediate reward
            result = env.step(action)
            total_reward = result.reward
            
        rewards.append(total_reward)
    return rewards

# ------------------------------------------------------------
# Main Training Script
# ------------------------------------------------------------
if __name__ == "__main__":
    # Load model (4-bit for memory efficiency)
    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/Qwen2.5-1.5B-Instruct",
        max_seq_length=1024,
        dtype=torch.float16,
        load_in_4bit=True,
    )

    # Attach trainable LoRA adapters on top of the 4-bit model
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Evaluate baselines
    print("Evaluating baseline policies...")
    rand_mean, rand_std = evaluate_policy(random_policy, num_episodes=5)
    print(f"Random policy: reward = {rand_mean:.2f} ± {rand_std:.2f}")
    sma_mean, sma_std = evaluate_policy(sma_crossover_policy, num_episodes=5)
    print(f"SMA crossover: reward = {sma_mean:.2f} ± {sma_std:.2f}")

    # Create realistic dataset with varied prompts
    from datasets import Dataset
    import random
    
    dummy_prompts = []
    data_env = DisciplinedTraderEnv()
    
    for i in range(50):   # 50 training examples
        seed_val = 42 + i
        obs = data_env.reset(task_id="easy", seed=seed_val)
        
        # Fast forward random amount of steps to gather a real state
        steps_to_advance = random.randint(10, 50)
        for _ in range(steps_to_advance):
            res = data_env.step(Action("do_nothing", 0))
            obs = res.observation
            
        prompt = (f"[SEED:{seed_val}][STEP:{steps_to_advance}]\n"
                  f"Observation: cash={data_env.cash:.0f}, value={data_env.account_value:.0f}, "
                  f"pos={data_env.position_shares}, price={obs.tf_1m.ohlcv.close:.2f}\n"
                  f"Regime: {obs.market_regime}, Pattern: {obs.tf_1m.pattern}\n"
                  "Generate an action in JSON: {\"action_type\": \"...\", \"amount_shares\": 0}")
        dummy_prompts.append(prompt)
    train_dataset = Dataset.from_dict({"prompt": dummy_prompts})

    # GRPO configuration
    training_args = GRPOConfig(
        output_dir="./trading_agent",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,  # Fix Unsloth warning: make this match bs * grad_accum
        learning_rate=1e-5,
        logging_steps=5,
        save_steps=20,
        max_prompt_length=128,
        max_completion_length=64,
        use_vllm=False,
        fp16=True,
        bf16=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func, 
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )

    print("\nStarting training...")
    trainer.train()

    # Save the trained LoRA adapter
    model.save_pretrained("./trained_trader_lora")
    tokenizer.save_pretrained("./trained_trader_lora")
    print("Trained model saved to ./trained_trader_lora")

    # Plot reward curve from training logs
    if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history'):
        logs = trainer.state.log_history
        # Look for 'episode_reward' or 'reward' in logs
        rewards = []
        for entry in logs:
            if 'episode_reward' in entry:
                rewards.append(entry['episode_reward'])
            elif 'reward' in entry and isinstance(entry['reward'], (int, float)):
                rewards.append(entry['reward'])
        if rewards:
            plt.plot(rewards)
            plt.xlabel("Training Step")
            plt.ylabel("Episode Reward")
            plt.title("Training Progress")
            plt.savefig("reward_curve.png")
            print("reward_curve.png saved")
        else:
            print("No reward entries found in training logs. Reward curve not generated.")
    else:
        print("Trainer state not accessible – reward curve not generated.")