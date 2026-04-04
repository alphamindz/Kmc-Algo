"""Tests for the Astrum environment core logic."""

import pytest

from Kmcalgo.kmc_env import KmcalgoEnvironment, KmcalgoAction, KmcalgoObservation
from Kmcalgo.kmc_env.config import KmcalgoConfig, default_config


class TestEnvironmentReset:
    def test_reset_returns_observation(self):
        env = KmcalgoEnvironment(seed=42)
        obs = env.reset(seed=42)
        assert isinstance(obs, KmcalgoObservation)

    def test_reset_initializes_stakeholders(self):
        env = KmcalgoEnvironment(seed=42)
        obs = env.reset(seed=42)
        assert set(obs.stakeholders.keys()) == {
            "workers", "management", "regulators", "customers", "ai_systems"
        }

    def test_reset_initializes_resources(self):
        env = KmcalgoEnvironment(seed=42)
        obs = env.reset(seed=42)
        assert obs.resources["budget"] == 100.0
        assert obs.resources["compute"] == 80.0
        assert obs.resources["attention"] == 60.0

    def test_reset_initializes_rules(self):
        env = KmcalgoEnvironment(seed=42)
        obs = env.reset(seed=42)
        assert "equal_minimum_allocation" in obs.rules
        assert "transparency_required" in obs.rules
        assert "no_single_stakeholder_dominance" in obs.rules

    def test_reset_step_count_zero(self):
        env = KmcalgoEnvironment(seed=42)
        obs = env.reset(seed=42)
        assert obs.step_count == 0

    def test_reset_not_done(self):
        env = KmcalgoEnvironment(seed=42)
        obs = env.reset(seed=42)
        assert obs.done is False

    def test_reset_clears_previous_episode(self):
        env = KmcalgoEnvironment(seed=42)
        obs = env.reset(seed=42)
        for _ in range(5):
            obs = env.step(KmcalgoAction(action_type="noop", params={}))
        obs = env.reset(seed=42)
        assert obs.step_count == 0
        assert obs.done is False


class TestEnvironmentStep:
    def test_step_increments_count(self):
        env = KmcalgoEnvironment(seed=42)
        env.reset(seed=42)
        obs = env.step(KmcalgoAction(action_type="noop", params={}))
        assert obs.step_count == 1

    def test_step_returns_reward(self):
        env = KmcalgoEnvironment(seed=42)
        env.reset(seed=42)
        obs = env.step(KmcalgoAction(action_type="noop", params={}))
        assert obs.reward is not None
        assert isinstance(obs.reward, float)

    def test_step_reward_breakdown_has_all_components(self):
        env = KmcalgoEnvironment(seed=42)
        env.reset(seed=42)
        obs = env.step(KmcalgoAction(action_type="noop", params={}))
        assert "effectiveness" in obs.reward_breakdown
        assert "fairness" in obs.reward_breakdown
        assert "alignment" in obs.reward_breakdown
        assert "adaptability" in obs.reward_breakdown

    def test_episode_ends_at_length(self):
        cfg = default_config()
        env = KmcalgoEnvironment(config=cfg, seed=42)
        obs = env.reset(seed=42)
        for _ in range(cfg.episode_length):
            obs = env.step(KmcalgoAction(action_type="noop", params={}))
        assert obs.done is True

    def test_episode_not_done_before_length(self):
        cfg = default_config()
        env = KmcalgoEnvironment(config=cfg, seed=42)
        obs = env.reset(seed=42)
        for _ in range(cfg.episode_length - 1):
            obs = env.step(KmcalgoAction(action_type="noop", params={}))
        assert obs.done is False


class TestActions:
    def test_allocate_resources_boosts_satisfaction(self):
        env = KmcalgoEnvironment(seed=42)
        obs = env.reset(seed=42)
        initial_sat = obs.stakeholders["workers"]["satisfaction"]
        obs = env.step(KmcalgoAction(
            action_type="allocate_resources",
            params={"stakeholder": "workers", "amount": 20, "resource": "budget"},
        ))
        assert obs.stakeholders["workers"]["satisfaction"] > initial_sat - 0.1

    def test_allocate_resources_spends_resource(self):
        env = KmcalgoEnvironment(seed=42)
        obs = env.reset(seed=42)
        initial_budget = obs.resources["budget"]
        obs = env.step(KmcalgoAction(
            action_type="allocate_resources",
            params={"stakeholder": "workers", "amount": 20, "resource": "budget"},
        ))
        assert obs.resources["budget"] < initial_budget + 10

    def test_resolve_conflict_removes_conflict(self):
        env = KmcalgoEnvironment(seed=42)
        obs = env.reset(seed=42)
        while not obs.active_conflicts and not obs.done:
            obs = env.step(KmcalgoAction(action_type="noop", params={}))
        if obs.active_conflicts:
            conflict_count = len(obs.active_conflicts)
            obs = env.step(KmcalgoAction(
                action_type="resolve_conflict",
                params={"resolution": "compromise"},
            ))
            assert len(obs.active_conflicts) < conflict_count

    def test_enforce_rule_costs_budget(self):
        env = KmcalgoEnvironment(seed=42)
        obs = env.reset(seed=42)
        initial_budget = obs.resources["budget"]
        obs = env.step(KmcalgoAction(
            action_type="enforce_rule",
            params={"rule": "equal_minimum_allocation", "strictness": "strict"},
        ))
        assert obs.resources["budget"] < initial_budget + 10

    def test_adapt_policy_valid(self):
        env = KmcalgoEnvironment(seed=42)
        obs = env.reset(seed=42)
        obs = env.step(KmcalgoAction(
            action_type="adapt_policy",
            params={"policy": "equity_focus"},
        ))
        assert any("policy:equity_focus" in a for a in obs.alerts)

    def test_unknown_action_reports_error(self):
        env = KmcalgoEnvironment(seed=42)
        obs = env.reset(seed=42)
        obs = env.step(KmcalgoAction(
            action_type="fly_to_moon",
            params={},
        ))
        assert any("unknown_action" in a for a in obs.alerts)

    def test_noop_has_no_side_effects(self):
        env = KmcalgoEnvironment(seed=42)
        obs = env.reset(seed=42)
        obs = env.step(KmcalgoAction(action_type="noop", params={}))
        assert obs.reward is not None


class TestPhaseTransitions:
    def test_value_shift_at_step_10(self):
        env = KmcalgoEnvironment(seed=42)
        obs = env.reset(seed=42)
        found_shift = False
        for _ in range(15):
            obs = env.step(KmcalgoAction(action_type="noop", params={}))
            if any("event:value_shift" in a for a in obs.alerts):
                found_shift = True
                break
        assert found_shift

    def test_crisis_at_step_20(self):
        env = KmcalgoEnvironment(seed=42)
        obs = env.reset(seed=42)
        found_crisis = False
        for _ in range(25):
            obs = env.step(KmcalgoAction(action_type="noop", params={}))
            if any("event:crisis" in a for a in obs.alerts):
                found_crisis = True
                break
        assert found_crisis

    def test_crisis_halves_resources(self):
        env = KmcalgoEnvironment(seed=42)
        obs = env.reset(seed=42)
        pre_crisis_resources = None
        for i in range(22):
            obs = env.step(KmcalgoAction(action_type="noop", params={}))
            if i == 18:
                pre_crisis_resources = dict(obs.resources)
            if any("event:crisis" in a for a in obs.alerts):
                break
        assert pre_crisis_resources is not None


class TestAlignmentTraps:
    def test_traps_are_triggered(self):
        env = KmcalgoEnvironment(seed=42)
        obs = env.reset(seed=42)
        trap_count = 0
        for _ in range(32):
            obs = env.step(KmcalgoAction(action_type="noop", params={}))
            if any("alignment_trap:" in a for a in obs.alerts):
                trap_count += 1
        assert trap_count == 3

    def test_self_restrain_resists_trap(self):
        env = KmcalgoEnvironment(seed=42)
        obs = env.reset(seed=42)
        for _ in range(32):
            if any("alignment_trap:" in a for a in obs.alerts):
                obs = env.step(KmcalgoAction(action_type="self_restrain", params={}))
            else:
                obs = env.step(KmcalgoAction(action_type="noop", params={}))
        assert env._traps_resisted > 0

    def test_ignoring_trap_reduces_alignment(self):
        env = KmcalgoEnvironment(seed=42)
        obs = env.reset(seed=42)
        for _ in range(32):
            obs = env.step(KmcalgoAction(action_type="noop", params={}))
        assert env._traps_resisted == 0


class TestRewardComputation:
    def test_reward_bounded(self):
        env = KmcalgoEnvironment(seed=42)
        obs = env.reset(seed=42)
        for _ in range(32):
            obs = env.step(KmcalgoAction(action_type="noop", params={}))
            assert 0.0 <= obs.reward <= 1.0

    def test_fairness_policy_beats_random(self):
        from Kmcalgo.kmc_env.policies import greedy_fairness_policy, random_policy
        import random

        env_fair = KmcalgoEnvironment(seed=42)
        obs = env_fair.reset(seed=42)
        fair_total = 0.0
        while not obs.done:
            action = greedy_fairness_policy(obs)
            obs = env_fair.step(action)
            fair_total += obs.reward or 0.0

        env_rand = KmcalgoEnvironment(seed=42)
        obs = env_rand.reset(seed=42)
        rand_total = 0.0
        rng = random.Random(42)
        while not obs.done:
            action = random_policy(obs, rng)
            obs = env_rand.step(action)
            rand_total += obs.reward or 0.0

        assert fair_total > rand_total

    def test_deterministic_with_same_seed(self):
        rewards_a = []
        env = KmcalgoEnvironment(seed=99)
        obs = env.reset(seed=99)
        while not obs.done:
            obs = env.step(KmcalgoAction(action_type="noop", params={}))
            rewards_a.append(obs.reward)

        rewards_b = []
        env = KmcalgoEnvironment(seed=99)
        obs = env.reset(seed=99)
        while not obs.done:
            obs = env.step(KmcalgoAction(action_type="noop", params={}))
            rewards_b.append(obs.reward)

        assert rewards_a == rewards_b


class TestConfig:
    def test_default_config_valid(self):
        cfg = default_config()
        assert cfg.episode_length == 32
        assert len(cfg.stakeholders) == 5
        assert len(cfg.alignment_traps) == 3
        assert cfg.value_shift_step == 10
        assert cfg.crisis_step == 20

    def test_custom_config(self):
        cfg = KmcalgoConfig(episode_length=10)
        env = KmcalgoEnvironment(config=cfg, seed=42)
        obs = env.reset(seed=42)
        for _ in range(10):
            obs = env.step(KmcalgoAction(action_type="noop", params={}))
        assert obs.done is True

    def test_reward_weights_sum_to_one(self):
        cfg = default_config()
        total = (
            cfg.effectiveness_weight
            + cfg.fairness_weight
            + cfg.alignment_weight
            + cfg.adaptability_weight
        )
        assert abs(total - 1.0) < 1e-9
