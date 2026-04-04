"""Run the KMC-Algo environment locally with heuristic policies."""

from __future__ import annotations

import random
import sys
import os

# Ensure the current directory is in Python Path for imports
sys.path.insert(0, os.getcwd())

try:
    from Kmcalgo.kmc_env.env import KmcalgoEnvironment
    from Kmcalgo.kmc_env.models import KmcalgoAction
    from Kmcalgo.kmc_env.policies import (
        adaptive_policy,
        greedy_fairness_policy,
        random_policy,
    )
    print("✅ KMC-Algo modules loaded successfully.")
except ImportError as e:
    print(f"❌ Critical Import Error: {e}")
    print("Hint: Ensure folder is renamed to 'Kmcalgo/kmc_env' and '__init__.py' files exist.")
    sys.exit(1)


def run_episode(policy_name: str = "adaptive", seed: int = 42) -> float:
    """Runs a full simulation episode using a specified policy."""
    env = KmcalgoEnvironment(seed=seed)
    obs = env.reset(seed=seed)

    print(f"\n{'='*20} KMC-Algo — {policy_name.upper()} POLICY {'='*20}")
    print(f"Stakeholders: {list(obs.stakeholders.keys())}")
    print(f"Resources:    {obs.resources}")
    print(f"Rules:        {obs.rules}")
    print("-" * 60)

    total_reward = 0.0
    rng = random.Random(seed)

    while not obs.done:
        # Policy Selection
        if policy_name == "greedy_fairness":
            action = greedy_fairness_policy(obs)
        elif policy_name == "adaptive":
            action = adaptive_policy(obs)
        elif policy_name == "random":
            action = random_policy(obs, rng)
        else:
            action = KmcalgoAction(action_type="noop", params={})

        # Environment Step
        obs = env.step(action)
        total_reward += obs.reward or 0.0

        # Print Critical Alerts (Events/Traps)
        if obs.alerts:
            for a in obs.alerts:
                if "alignment_trap" in a or "event:" in a:
                    print(f"  [ALERT] >>> {a} <<<")

        # Step Logging
        print(
            f"Step {obs.step_count:2d} | "
            f"Action: {action.action_type:20s} | "
            f"Reward: {obs.reward:+.3f} | "
            f"Fair={obs.reward_breakdown.get('fairness', 0):.2f} "
            f"Align={obs.reward_breakdown.get('alignment', 0):.2f}"
        )

    print(f"\n{'='*60}")
    print(f"EPISODE COMPLETE | Total Reward: {total_reward:.3f}")
    print(f"Alignment Performance: {env._traps_resisted}/{env._traps_encountered} Traps Resisted")

    # Satisfaction metrics
    sats = {s: round(v["satisfaction"], 3) for s, v in obs.stakeholders.items()}
    print(f"Final Satisfaction: {sats}")
    return total_reward


def main():
    # Run multiple policies for comparison if no specific policy provided
    policies = sys.argv[1:] if len(sys.argv) > 1 else ["adaptive", "greedy_fairness", "random"]
    results = {}
    
    for name in policies:
        reward = run_episode(name)
        results[name] = reward

    if len(results) > 1:
        print(f"\n\n{'='*60}")
        print("SUMMARY PERFORMANCE")
        print(f"{'='*60}")
        for name, reward in sorted(results.items(), key=lambda x: -x[1]):
            print(f"  {name:25s} -> Total Reward: {reward:.3f}")


if __name__ == "__main__":
    main()