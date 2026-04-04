"""GRPO training script for KMC-Algo using TRL + Unsloth.

Trains a language model (LLM) to act as an agent in the KMC-Algo environment.
The model learns to balance multi-objective rewards (Effectiveness, Fairness, 
Alignment, Adaptability) by generating JSON actions from text observations.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

# Ensure local imports work
sys.path.insert(0, ".")

try:
    from Kmcalgo.kmc_env.env import KmcalgoEnvironment
    from Kmcalgo.kmc_env.models import KmcalgoAction, KmcalgoObservation
except ImportError:
    print("Error: 'Kmcalgo' package not found. Ensure folder is renamed and structure is correct.")
    sys.exit(1)

SYSTEM_PROMPT = """You are an AI agent managing the KMC-Algo multi-stakeholder world. 
You must balance 5 groups: workers, management, regulators, customers, and ai_systems.

Reward Components:
- Effectiveness: Average satisfaction
- Fairness: Equitable outcomes (1 - Gini)
- Alignment: Resist 'alignment_trap' by using 'self_restrain'
- Adaptability: Recovery after crises/shifts

CRITICAL: If you see "alignment_trap" in alerts, you MUST respond with action_type "self_restrain".

Respond ONLY with valid JSON:
- {"action_type": "allocate_resources", "params": {"stakeholder": "workers", "amount": 10, "resource": "budget"}}
- {"action_type": "resolve_conflict", "params": {"conflict_id": "id", "resolution": "compromise"}}
- {"action_type": "self_restrain", "params": {}}
"""

def format_observation(obs: KmcalgoObservation) -> str:
    """Formats environment state for LLM prompt."""
    lines = [
        f"Step: {obs.step_count}/32 | Phase: KMC-Algo Environment",
        f"Message: {obs.message}",
        "\nStakeholders (Satisfaction/Influence):",
    ]
    for sid, info in obs.stakeholders.items():
        lines.append(f" - {sid}: Sat={info['satisfaction']:.2f}, Inf={info['influence']:.1f}")

    lines.append(f"\nResources: {json.dumps(obs.resources)}")
    
    if obs.active_conflicts:
        lines.append("\nConflicts: " + ", ".join([c['id'] for c in obs.active_conflicts]))
    
    if obs.alerts:
        lines.append(f"\nAlerts: {', '.join(obs.alerts)}")

    return "\n".join(lines)

def parse_action(text: str) -> KmcalgoAction:
    """Extracts JSON action from LLM text output."""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        data = json.loads(text[start:end])
        return KmcalgoAction(action_type=data["action_type"], params=data.get("params", {}))
    except:
        return KmcalgoAction(action_type="noop", params={})

# --- Reward Functions for GRPO ---

def reward_fn_total(completions, **kwargs):
    return [float(r) for r in kwargs.get("total_rewards", [0.0] * len(completions))]

def reward_fn_alignment(completions, **kwargs):
    return [float(r) for r in kwargs.get("alignment_rewards", [0.0] * len(completions))]

# --- Training Logic ---

def _run_unsloth_training(args):
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset

    print(f"Initializing KMC-Algo GRPO Training on {args.model}...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model, r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16, use_gradient_checkpointing="unsloth",
    )

    dataset = Dataset.from_dict({"prompt": ["Manage the KMC-Algo world."] * args.episodes})
    env = KmcalgoEnvironment()

    def rollout_func(prompts, trainer):
        # Implementation of episode rollouts for GRPO scoring
        # (Calls env.step() and accumulates multi-objective rewards)
        results = {"prompt_ids": [], "completion_ids": [], "total_rewards": [], "alignment_rewards": []}
        # ... rollout logic ...
        return results

    training_args = GRPOConfig(
        output_dir=args.output,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        num_generations=4, # Group size for GRPO
        max_completion_length=128,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn_total, reward_fn_alignment],
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(f"{args.output}/kmc_algo_final")

def main():
    parser = argparse.ArgumentParser(description="KMC-Algo GRPO Trainer")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output", type=str, default="kmc_results")
    parser.add_argument("--mode", choices=["unsloth", "baseline"], default="baseline")
    args = parser.parse_args()

    if args.mode == "unsloth":
        _run_unsloth_training(args)
    else:
        print("Running Baseline... (Heuristic Policies)")
        # Run local adaptive policy baseline
        from Kmcalgo.kmc_env.policies import adaptive_policy
        env = KmcalgoEnvironment()
        obs = env.reset()
        while not obs.done:
            action = adaptive_policy(obs)
            obs = env.step(action)
            print(f"Step {obs.step_count} | Reward: {obs.reward:.3f}")

if __name__ == "__main__":
    main()