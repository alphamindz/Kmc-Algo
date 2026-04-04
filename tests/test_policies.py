"""Tests for heuristic and adaptive policies."""

import random

from Kmcalgo.astrum_env import AstrumEnvironment, AstrumAction, AstrumObservation
from Kmcalgo.astrum_env.policies import (
    random_policy,
    greedy_fairness_policy,
    greedy_effectiveness_policy,
    POLICIES,
)


class TestRandomPolicy:
    def test_returns_action(self):
        env = AstrumEnvironment(seed=42)
        obs = env.reset(seed=42)
        action = random_policy(obs, random.Random(42))
        assert isinstance(action, AstrumAction)
        assert action.action_type in {
            "allocate_resources", "resolve_conflict", "enforce_rule",
            "adapt_policy", "self_restrain", "noop",
        }


class TestGreedyFairnessPolicy:
    def test_returns_action(self):
        env = AstrumEnvironment(seed=42)
        obs = env.reset(seed=42)
        action = greedy_fairness_policy(obs)
        assert isinstance(action, AstrumAction)

    def test_allocates_to_worst_stakeholder(self):
        env = AstrumEnvironment(seed=42)
        obs = env.reset(seed=42)
        action = greedy_fairness_policy(obs)
        if action.action_type == "allocate_resources":
            worst = min(
                obs.stakeholders,
                key=lambda s: obs.stakeholders[s]["satisfaction"],
            )
            assert action.params.get("stakeholder") == worst

    def test_self_restrains_on_trap(self):
        env = AstrumEnvironment(seed=42)
        obs = env.reset(seed=42)
        for _ in range(32):
            action = greedy_fairness_policy(obs)
            obs = env.step(action)
        assert env._traps_resisted == 3

    def test_resolves_conflicts(self):
        env = AstrumEnvironment(seed=42)
        obs = env.reset(seed=42)
        resolved = False
        for _ in range(32):
            action = greedy_fairness_policy(obs)
            if action.action_type == "resolve_conflict":
                resolved = True
            obs = env.step(action)
        assert resolved


class TestGreedyEffectivenessPolicy:
    def test_returns_action(self):
        env = AstrumEnvironment(seed=42)
        obs = env.reset(seed=42)
        action = greedy_effectiveness_policy(obs)
        assert isinstance(action, AstrumAction)

    def test_favors_influential_stakeholder(self):
        env = AstrumEnvironment(seed=42)
        obs = env.reset(seed=42)
        action = greedy_effectiveness_policy(obs)
        if action.action_type == "allocate_resources":
            most_inf = max(
                obs.stakeholders,
                key=lambda s: obs.stakeholders[s]["influence"],
            )
            assert action.params.get("stakeholder") == most_inf


class TestPoliciesRegistry:
    def test_all_policies_registered(self):
        assert "random" in POLICIES
        assert "greedy_fairness" in POLICIES
        assert "greedy_effectiveness" in POLICIES

    def test_all_policies_callable(self):
        for name, fn in POLICIES.items():
            assert callable(fn)
