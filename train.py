"""Training script for KMC-Algo on GPU.

Runs multi-episode training with exploration annealing, tracking reward curves,
alignment trap resistance, and fairness over time. Optimized for GPU execution.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

# Final Package Imports
from Kmcalgo.kmc_env.env import KmcalgoEnvironment
from Kmcalgo.kmc_env.models import KmcalgoAction
from Kmcalgo.kmc_env.policies import (
    adaptive_policy,
    greedy_fairness_policy,
    random_policy,
)

@dataclass
class EpisodeResult:
    episode: int
    seed: int
    policy: str
    total_reward: float
    traps_resisted: int
    traps_encountered: int
    final_fairness: float
    final_effectiveness: float
    final_alignment: float
    final_adaptability: float
    final_satisfaction: dict = field(default_factory=dict)
    elapsed_ms: float = 0.0

@dataclass
class TrainingRun:
    run_id: str
    episodes: int
    results: List[EpisodeResult] = field(default_factory=list)
    total_elapsed_s: float = 0.0

def run_episode(
    policy_name: str,
    seed: int,
    episode_num: int,
    exploration_rate: float = 0.0,
) -> EpisodeResult:
    """Simulates a single episode and returns performance metrics."""
    env = KmcalgoEnvironment(seed=seed)
    obs = env.reset(seed=seed)
    rng = random.Random(seed)

    total_reward = 0.0
    last_breakdown = {}
    t0 = time.perf_counter()

    while not obs.done:
        # Policy Selection Logic
        if policy_name == "adaptive":
            action = adaptive_policy(obs)
        elif policy_name == "greedy_fairness":
            action = greedy_fairness_policy(obs)
        elif policy_name == "random":
            action = random_policy(obs, rng)
        elif policy_name == "adaptive_explore":
            # Exploration Annealing Logic
            if rng.random() < exploration_rate:
                action = random_policy(obs, rng)
            else:
                action = adaptive_policy(obs)
        else:
            action = KmcalgoAction(action_type="noop", params={})

        obs = env.step(action)
        total_reward += obs.reward or 0.0
        last_breakdown = obs.reward_breakdown

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return EpisodeResult(
        episode=episode_num,
        seed=seed,
        policy=policy_name,
        total_reward=round(total_reward, 4),
        traps_resisted=env._traps_resisted,
        traps_encountered=env._traps_encountered,
        final_fairness=round(last_breakdown.get("fairness", 0), 4),
        final_effectiveness=round(last_breakdown.get("effectiveness", 0), 4),
        final_alignment=round(last_breakdown.get("alignment", 0), 4),
        final_adaptability=round(last_breakdown.get("adaptability", 0), 4),
        final_satisfaction={
            s: round(v["satisfaction"], 4) for s, v in obs.stakeholders.items()
        },
        elapsed_ms=round(elapsed_ms, 2),
    )

def train(n_episodes: int = 200, output_dir: str = "kmc_results") -> TrainingRun:
    """Executes the training loop with reward tracking."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    run_id = f"kmc_algo_{int(time.time())}"

    print(f"{'='*70}")
    print(f" KMC-Algo Training Run: {run_id}")
    print(f"Episodes: {n_episodes}")
    print(f"{'='*70}\n")

    training_run = TrainingRun(run_id=run_id, episodes=n_episodes)

    # Exploration Annealing Schedule (Start high, end low)
    exploration_rates = [
        max(0.05, 0.8 - (0.75 * i / max(1, n_episodes - 1)))
        for i in range(n_episodes)
    ]

    t_start = time.perf_counter()

    for ep in range(n_episodes):
        result = run_episode(
            policy_name="adaptive_explore",
            seed=ep,
            episode_num=ep,
            exploration_rate=exploration_rates[ep],
        )
        training_run.results.append(result)

        if ep % 10 == 0 or ep == n_episodes - 1:
            recent = training_run.results[max(0, ep - 9):]
            avg_reward = sum(r.total_reward for r in recent) / len(recent)
            avg_traps = sum(r.traps_resisted for r in recent) / len(recent)
            print(
                f"Ep {ep:4d}/{n_episodes} | Explore: {exploration_rates[ep]:.2f} | "
                f"Reward: {avg_reward:.3f} | Traps: {avg_traps:.1f}/3 | {result.elapsed_ms:.1f}ms"
            )

    training_run.total_elapsed_s = round(time.perf_counter() - t_start, 2)
    print(f"\nTraining Complete in {training_run.total_elapsed_s}s")

    # Baseline Comparisons
    print("\n--- Baseline Comparison ---")
    for b_name in ["adaptive", "greedy_fairness", "random"]:
        res = run_episode(policy_name=b_name, seed=42, episode_num=-1)
        print(f" {b_name:20s}: Reward={res.total_reward:.3f}, Traps={res.traps_resisted}/3")

    # Final Save
    results_path = os.path.join(output_dir, f"{run_id}.json")
    with open(results_path, "w") as f:
        json.dump(asdict(training_run), f, indent=2)
    print(f"\nFull log saved to: {results_path}")

    return training_run

def main():
    parser = argparse.ArgumentParser(description="KMC-Algo Training Loop")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--output", type=str, default="kmc_results")
    args = parser.parse_args()

    train(n_episodes=args.episodes, output_dir=args.output)

if __name__ == "__main__":
    main()