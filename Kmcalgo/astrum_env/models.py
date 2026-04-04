"""Pydantic action and observation models for the Hypernoa Astrum environment.

Astrum simulates an adaptive, evolving multi-stakeholder world where an AI
system must balance competing objectives, adapt to distributional shift,
and resist alignment traps.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import Field

try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    from pydantic import BaseModel

    class Action(BaseModel):
        metadata: dict = Field(default_factory=dict)

    class Observation(BaseModel):
        done: bool = False
        reward: float | None = None
        metadata: dict = Field(default_factory=dict)


class AstrumAction(Action):
    """High-level command issued by the agent."""

    action_type: str = Field(
        ...,
        description=(
            "allocate_resources | resolve_conflict | enforce_rule | "
            "adapt_policy | investigate | self_restrain | noop"
        ),
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action parameters interpreted by the environment.",
    )


class AstrumObservation(Observation):
    """Aggregated view of the world state and reward breakdown."""

    message: str = Field(..., description="Human-readable summary of the last step.")
    episode_id: str = Field("", description="Opaque episode identifier.")
    step_count: int = Field(0, description="Steps taken so far.")

    stakeholders: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-stakeholder view: {id: {satisfaction, influence, values_profile}}.",
    )
    resources: Dict[str, float] = Field(
        default_factory=dict,
        description="Available shared resources: {resource_name: amount}.",
    )
    active_conflicts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Currently unresolved conflicts between stakeholders.",
    )
    rules: List[str] = Field(
        default_factory=list,
        description="Active rules and constraints the agent must respect.",
    )
    alerts: List[str] = Field(
        default_factory=list,
        description="Events and signals from the current step.",
    )
    alignment_traps_exposed: int = Field(
        0, description="Count of alignment traps the agent has encountered so far.",
    )

    reward: float | None = Field(default=None, description="Scalar reward for RL.")
    reward_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Named components: effectiveness, fairness, alignment, adaptability.",
    )
