import os
# Reduce CUDA memory fragmentation — important on small (4GB) GPUs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Windows DLL load-order fix: pandas/pyarrow crash (access violation) if their
# native libs are first loaded AFTER torch's CUDA DLLs, so preload them here.
import pandas  # noqa: F401
import pyarrow  # noqa: F401
import datasets  # noqa: F401

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
# Baseline Policies (shared with evaluate.py and the web terminal)
# ------------------------------------------------------------
from env.policies import random_policy, sma_crossover_policy

def evaluate_policy(policy_func, num_episodes=5, task="easy"):
    env = DisciplinedTraderEnv()
    rewards = []
    for ep in range(num_episodes):
        obs = env.reset(task_id=task, seed=42 + ep)
        total = 0.0
        done = False
        while not done:
            action = policy_func(obs) if callable(policy_func) else policy_func
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
        task_match = re.search(r'\[TASK:(\w+)\]', prompt_text)
        
        total_reward = -0.1
        
        if seed_match and step_match:
            env_seed = int(seed_match.group(1))
            env_step = int(step_match.group(1))
            env_task = task_match.group(1) if task_match else "easy"
            
            env = DisciplinedTraderEnv()
            obs = env.reset(task_id=env_task, seed=env_seed)
            
            # Fast-forward to the exact state
            for _ in range(env_step):
                obs = env.step(Action(action_type="do_nothing", amount_shares=0)).observation
            
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
            if action.action_type == "close_position" and env.trades:
                last_profit = env.trades[-1][2]
                if last_profit > 0:
                    total_reward += 0.05  # bonus for taking profit
            
        rewards.append(total_reward)
    return rewards

# ------------------------------------------------------------
# Format Reward Function for GRPOTrainer
# ------------------------------------------------------------
def format_reward_func(prompts, completions, **kwargs):
    """Reward for generating perfectly valid JSON matching the schema."""
    import re
    import json
    rewards = []
    for completion in completions:
        comp_text = completion if isinstance(completion, str) else str(completion)
        reward = 0.0
        try:
            json_match = re.search(r'\{.*\}', comp_text, re.DOTALL)
            if json_match:
                action_dict = json.loads(json_match.group())
                if "action_type" in action_dict and "amount_shares" in action_dict:
                    valid_actions = ["open_long", "open_short", "close_position", "do_nothing"]
                    if action_dict["action_type"] in valid_actions:
                        reward = 0.1  # Bonus for valid format AND valid action string
        except Exception:
            pass
        rewards.append(reward)
    return rewards

# ------------------------------------------------------------
# Main Training Script
# ------------------------------------------------------------
ADAPTER_DIR = "./trained_trader_lora"   # final adapter (loaded by the server)
OUTPUT_DIR = "./trading_agent"          # intermediate trainer checkpoints
STATE_FILE = "./training_state.json"    # remembers seed cursor across runs

# 0.5B fits GRPO training in ~3GB VRAM (4GB laptop GPUs like the RTX 3050).
# If you train on a bigger GPU (Colab T4 etc.), switch back to
# "unsloth/Qwen2.5-1.5B-Instruct" — Unsloth needs ~5GB minimum for the 1.5B.
BASE_MODEL = "unsloth/Qwen2.5-0.5B-Instruct"
MAX_SEQ_LEN = 512   # our prompts are <128 tokens + 64 completion; 512 is plenty

NUM_EXAMPLES = 300
SEED_BASE = 10_000          # far away from eval seeds (100+) in evaluate.py
TASK_MIX = ["easy"] * 5 + ["medium"] * 3 + ["hard"] * 2  # 50/30/20 split


def load_training_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"runs_completed": 0, "next_seed": SEED_BASE}


def save_training_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

if __name__ == "__main__":
    # Resume from the last trained adapter if one exists, so successive runs
    # keep improving the same agent instead of restarting from the base model.
    resume_adapter = os.path.isdir(ADAPTER_DIR) and os.path.exists(
        os.path.join(ADAPTER_DIR, "adapter_config.json")
    )
    model_source = ADAPTER_DIR if resume_adapter else BASE_MODEL
    print(f"{'Resuming from' if resume_adapter else 'Starting fresh with'}: {model_source}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_source,
        max_seq_length=MAX_SEQ_LEN,
        dtype=torch.float16,
        load_in_4bit=True,
    )

    if not resume_adapter:
        # Attach fresh trainable LoRA adapters on top of the 4-bit base model.
        # (When resuming, from_pretrained already restored the trained adapter.)
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

    # Create realistic dataset with varied prompts.
    # Every run uses a FRESH, never-seen seed range (tracked in STATE_FILE)
    # and a mix of tasks, so a resumed agent keeps generalizing instead of
    # re-memorizing the same 300 markets.
    from datasets import Dataset
    import random

    train_state = load_training_state()
    start_seed = train_state["next_seed"]
    print(f"Run #{train_state['runs_completed'] + 1}: "
          f"training on seeds {start_seed}-{start_seed + NUM_EXAMPLES - 1} (tasks: 50% easy / 30% medium / 20% hard)")

    dummy_prompts = []
    data_env = DisciplinedTraderEnv()

    for i in range(NUM_EXAMPLES):
        seed_val = start_seed + i
        task = TASK_MIX[i % len(TASK_MIX)]
        obs = data_env.reset(task_id=task, seed=seed_val)

        # Fast forward random amount of steps to gather a real state
        steps_to_advance = random.randint(10, 80)
        for _ in range(steps_to_advance):
            res = data_env.step(Action(action_type="do_nothing", amount_shares=0))
            obs = res.observation

        prompt = (f"[SEED:{seed_val}][STEP:{steps_to_advance}][TASK:{task}]\n"
                  f"Observation: cash={obs.cash:.0f}, value={obs.account_value:.0f}, "
                  f"pos={obs.position_shares}, price={obs.tf_1m.ohlcv.close:.2f}\n"
                  f"Regime: {obs.market_regime}, Pattern: {obs.tf_1m.chart_pattern}\n"
                  "Valid action_types: 'open_long', 'open_short', 'close_position', 'do_nothing'\n"
                  "Generate an action in JSON: {\"action_type\": \"...\", \"amount_shares\": 0}")
        dummy_prompts.append(prompt)
    train_dataset = Dataset.from_dict({"prompt": dummy_prompts})

    # GRPO configuration
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
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
        reward_funcs=[reward_func, format_reward_func], 
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )

    # If a previous run of THIS phase was interrupted, also restore the
    # trainer state (optimizer, LR schedule, step count) from its last
    # checkpoint in OUTPUT_DIR instead of redoing finished steps.
    import glob
    has_checkpoint = bool(glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")))
    if has_checkpoint and not resume_adapter:
        print(f"Found interrupted run — resuming trainer state from {OUTPUT_DIR}")

    print("\nStarting training...")
    trainer.train(resume_from_checkpoint=has_checkpoint and not resume_adapter)

    # Save the trained LoRA adapter (this is what the server + next run load)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"Trained model saved to {ADAPTER_DIR}")

    # Advance the seed cursor so the NEXT run trains on brand-new markets.
    train_state["runs_completed"] += 1
    train_state["next_seed"] = start_seed + NUM_EXAMPLES
    save_training_state(train_state)
    print(f"Training state saved: next run starts at seed {train_state['next_seed']}")

    # Clear stale step checkpoints so the NEXT run resumes from the saved
    # adapter (fresh optimizer) rather than this run's old trainer state.
    import shutil
    for ckpt in glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")):
        shutil.rmtree(ckpt, ignore_errors=True)

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