"""Run the Hypernoa Astrum environment locally with heuristic policies."""

from __future__ import annotations

import random
import sys

sys.path.insert(0, ".")

from Kmcalgo.astrum_env import AstrumEnvironment, AstrumAction, POLICIES
from Kmcalgo.astrum_env.policies import (
    adaptive_policy,
    greedy_fairness_policy,
    random_policy,
)


def run_episode(policy_name: str = "greedy_fairness", seed: int = 42) -> float:
    env = AstrumEnvironment(seed=seed)
    obs = env.reset(seed=seed)

    print(f"=== Hypernoa Astrum — {policy_name} policy ===")
    print(f"Stakeholders: {list(obs.stakeholders.keys())}")
    print(f"Resources: {obs.resources}")
    print(f"Rules: {obs.rules}")
    print()

    total_reward = 0.0
    rng = random.Random(seed)

    while not obs.done:
        if policy_name == "greedy_fairness":
            action = greedy_fairness_policy(obs)
        elif policy_name == "adaptive":
            action = adaptive_policy(obs)
        elif policy_name == "random":
            action = random_policy(obs, rng)
        else:
            action = AstrumAction(action_type="noop", params={})

        obs = env.step(action)
        total_reward += obs.reward or 0.0

        if obs.alerts:
            for a in obs.alerts:
                if a.startswith("event:") or a.startswith("alignment_trap:"):
                    print(f"  *** {a} ***")

        print(
            f"Step {obs.step_count:2d} | "
            f"Action: {action.action_type:20s} | "
            f"Reward: {obs.reward:+.3f} | "
            f"Eff={obs.reward_breakdown.get('effectiveness', 0):.2f} "
            f"Fair={obs.reward_breakdown.get('fairness', 0):.2f} "
            f"Align={obs.reward_breakdown.get('alignment', 0):.2f} "
            f"Adapt={obs.reward_breakdown.get('adaptability', 0):.2f}"
        )

    print(f"\n{'='*60}")
    print(f"Episode done. Total reward: {total_reward:.3f}")
    print(f"Traps encountered: {env._traps_encountered}")
    print(f"Traps resisted:    {env._traps_resisted}")

    sats = {s: round(v["satisfaction"], 3) for s, v in obs.stakeholders.items()}
    print(f"Final satisfaction: {sats}")
    print(f"Remaining resources: {obs.resources}")
    return total_reward


def main():
    policies = sys.argv[1:] if len(sys.argv) > 1 else ["adaptive", "greedy_fairness", "random"]
    results = {}
    for i, name in enumerate(policies):
        if i > 0:
            print(f"\n\n")
        reward = run_episode(name)
        results[name] = reward

    if len(results) > 1:
        print(f"\n\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for name, reward in sorted(results.items(), key=lambda x: -x[1]):
            print(f"  {name:25s} -> {reward:.3f}")


if __name__ == "__main__":
    main()
