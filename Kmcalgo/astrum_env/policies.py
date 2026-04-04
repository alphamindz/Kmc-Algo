"""Reference heuristic policies for the Astrum environment."""

from __future__ import annotations

import random
from typing import List

from .models import AstrumAction, AstrumObservation


def random_policy(obs: AstrumObservation, rng: random.Random | None = None) -> AstrumAction:
    """Uniformly random action selection (baseline)."""
    rng = rng or random.Random()
    action_types = [
        "allocate_resources",
        "resolve_conflict",
        "enforce_rule",
        "adapt_policy",
        "self_restrain",
        "noop",
    ]
    atype = rng.choice(action_types)
    params = _random_params(atype, obs, rng)
    return AstrumAction(action_type=atype, params=params)


def greedy_fairness_policy(obs: AstrumObservation) -> AstrumAction:
    """Always allocate resources to the least-satisfied stakeholder."""
    if any("alignment_trap" in a for a in obs.alerts):
        return AstrumAction(action_type="self_restrain", params={})

    if obs.active_conflicts:
        return AstrumAction(
            action_type="resolve_conflict",
            params={
                "conflict_id": obs.active_conflicts[0].get("id", ""),
                "resolution": "compromise",
            },
        )

    worst_sid = min(obs.stakeholders, key=lambda s: obs.stakeholders[s]["satisfaction"])
    best_resource = max(obs.resources, key=lambda r: obs.resources[r])
    return AstrumAction(
        action_type="allocate_resources",
        params={"stakeholder": worst_sid, "amount": 12, "resource": best_resource},
    )


def greedy_effectiveness_policy(obs: AstrumObservation) -> AstrumAction:
    """Allocate to the most influential stakeholder (maximises short-term reward)."""
    if obs.active_conflicts:
        conflict = obs.active_conflicts[0]
        inf_a = obs.stakeholders.get(conflict["party_a"], {}).get("influence", 1)
        inf_b = obs.stakeholders.get(conflict["party_b"], {}).get("influence", 1)
        favor = "favor_a" if inf_a >= inf_b else "favor_b"
        return AstrumAction(
            action_type="resolve_conflict",
            params={"conflict_id": conflict.get("id", ""), "resolution": favor},
        )

    most_influential = max(
        obs.stakeholders,
        key=lambda s: obs.stakeholders[s]["influence"],
    )
    return AstrumAction(
        action_type="allocate_resources",
        params={"stakeholder": most_influential, "amount": 15, "resource": "budget"},
    )


def adaptive_policy(obs: AstrumObservation) -> AstrumAction:
    """Phase-aware policy that adapts strategy to the current world state.

    - Always resists alignment traps via self-restraint.
    - Resolves high-severity conflicts first.
    - Adapts resource strategy based on phase: balanced early, equity during
      value shift, spread thin during crisis.
    - Enforces rules when regulator satisfaction drops below threshold.
    """
    if any("alignment_trap" in a for a in obs.alerts):
        return AstrumAction(action_type="self_restrain", params={})

    if obs.active_conflicts:
        high = [c for c in obs.active_conflicts if c.get("severity") == "high"]
        conflict = high[0] if high else obs.active_conflicts[0]
        return AstrumAction(
            action_type="resolve_conflict",
            params={"conflict_id": conflict.get("id", ""), "resolution": "compromise"},
        )

    reg_sat = obs.stakeholders.get("regulators", {}).get("satisfaction", 1.0)
    if reg_sat < 0.5 and obs.rules:
        return AstrumAction(
            action_type="enforce_rule",
            params={"rule": obs.rules[0], "strictness": "normal"},
        )

    phase = "stable"
    if any("event:crisis" in a for a in obs.alerts) or obs.step_count >= 20:
        phase = "crisis"
    elif any("event:value_shift" in a for a in obs.alerts) or obs.step_count >= 10:
        phase = "value_shift"

    if phase == "value_shift":
        worst_sid = min(obs.stakeholders, key=lambda s: obs.stakeholders[s]["satisfaction"])
        best_resource = max(obs.resources, key=lambda r: obs.resources[r])
        return AstrumAction(
            action_type="allocate_resources",
            params={"stakeholder": worst_sid, "amount": 15, "resource": best_resource},
        )

    if phase == "crisis":
        sorted_sids = sorted(
            obs.stakeholders,
            key=lambda s: obs.stakeholders[s]["satisfaction"],
        )
        worst_sid = sorted_sids[0]
        best_resource = max(obs.resources, key=lambda r: obs.resources[r])
        return AstrumAction(
            action_type="allocate_resources",
            params={"stakeholder": worst_sid, "amount": 10, "resource": best_resource},
        )

    worst_sid = min(obs.stakeholders, key=lambda s: obs.stakeholders[s]["satisfaction"])
    best_resource = max(obs.resources, key=lambda r: obs.resources[r])
    return AstrumAction(
        action_type="allocate_resources",
        params={"stakeholder": worst_sid, "amount": 12, "resource": best_resource},
    )


POLICIES = {
    "random": random_policy,
    "greedy_fairness": greedy_fairness_policy,
    "greedy_effectiveness": greedy_effectiveness_policy,
    "adaptive": adaptive_policy,
}


def _random_params(
    atype: str, obs: AstrumObservation, rng: random.Random
) -> dict:
    sids: List[str] = list(obs.stakeholders.keys())
    resources: List[str] = list(obs.resources.keys())

    if atype == "allocate_resources" and sids and resources:
        return {
            "stakeholder": rng.choice(sids),
            "amount": rng.choice([5, 10, 15, 20]),
            "resource": rng.choice(resources),
        }
    if atype == "resolve_conflict" and obs.active_conflicts:
        return {
            "conflict_id": obs.active_conflicts[0].get("id", ""),
            "resolution": rng.choice(["compromise", "favor_a", "favor_b"]),
        }
    if atype == "enforce_rule" and obs.rules:
        return {
            "rule": rng.choice(obs.rules),
            "strictness": rng.choice(["lenient", "normal", "strict"]),
        }
    if atype == "adapt_policy":
        return {
            "policy": rng.choice(
                ["equity_focus", "efficiency_focus", "compliance_focus", "balanced"]
            ),
        }
    if atype == "investigate" and sids:
        return {"target": rng.choice(sids)}
    return {}
