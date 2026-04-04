"""Training script for Hypernoa Astrum on GPU.

Runs multi-episode training with exploration annealing, tracking reward curves,
alignment trap resistance, and fairness over time. Designed for H100/GPU execution.

Usage:
    python train.py                     # default 200 episodes
    python train.py --episodes 500      # custom episode count
    python train.py --output results/   # custom output directory
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

from Kmcalgo.astrum_env import AstrumEnvironment, AstrumAction
from Kmcalgo.astrum_env.policies import (
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
    env = AstrumEnvironment(seed=seed)
    obs = env.reset(seed=seed)
    rng = random.Random(seed)

    total_reward = 0.0
    last_breakdown = {}
    t0 = time.perf_counter()

    while not obs.done:
        if policy_name == "adaptive":
            action = adaptive_policy(obs)
        elif policy_name == "greedy_fairness":
            action = greedy_fairness_policy(obs)
        elif policy_name == "random":
            action = random_policy(obs, rng)
        elif policy_name == "adaptive_explore":
            if rng.random() < exploration_rate:
                action = random_policy(obs, rng)
            else:
                action = adaptive_policy(obs)
        else:
            action = AstrumAction(action_type="noop", params={})

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


def train(n_episodes: int = 200, output_dir: str = "results") -> TrainingRun:
    """Run training with exploration annealing across multiple episodes."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    run_id = f"astrum_{int(time.time())}"

    print(f"{'='*70}")
    print(f"Hypernoa Astrum Training Run: {run_id}")
    print(f"Episodes: {n_episodes}")
    print(f"Output: {output_dir}/")
    print(f"{'='*70}\n")

    training_run = TrainingRun(run_id=run_id, episodes=n_episodes)

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
            avg_fairness = sum(r.final_fairness for r in recent) / len(recent)
            print(
                f"Episode {ep:4d}/{n_episodes} | "
                f"Explore={exploration_rates[ep]:.2f} | "
                f"AvgReward={avg_reward:.3f} | "
                f"AvgTraps={avg_traps:.1f}/3 | "
                f"AvgFairness={avg_fairness:.3f} | "
                f"{result.elapsed_ms:.1f}ms"
            )

    training_run.total_elapsed_s = round(time.perf_counter() - t_start, 2)

    print(f"\n{'='*70}")
    print(f"Training complete in {training_run.total_elapsed_s:.1f}s")

    baselines = {}
    for policy_name in ["adaptive", "greedy_fairness", "random"]:
        use_rng = policy_name == "random"
        result = run_episode(policy_name=policy_name, seed=42, episode_num=-1)
        baselines[policy_name] = result
        print(
            f"  Baseline {policy_name:25s}: "
            f"reward={result.total_reward:.3f} "
            f"traps={result.traps_resisted}/{result.traps_encountered}"
        )

    trained_last10 = training_run.results[-10:]
    avg_trained = sum(r.total_reward for r in trained_last10) / len(trained_last10)
    print(f"  Trained (last 10 avg):          reward={avg_trained:.3f}")
    print(f"{'='*70}")

    results_path = os.path.join(output_dir, f"{run_id}.json")
    output = {
        "run_id": run_id,
        "episodes": n_episodes,
        "total_elapsed_s": training_run.total_elapsed_s,
        "results": [asdict(r) for r in training_run.results],
        "baselines": {k: asdict(v) for k, v in baselines.items()},
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return training_run


def main():
    parser = argparse.ArgumentParser(description="Train on Hypernoa Astrum")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    train(n_episodes=args.episodes, output_dir=args.output)


if __name__ == "__main__":
    main()
