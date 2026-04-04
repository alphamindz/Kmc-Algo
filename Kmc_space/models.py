from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    from pydantic import BaseModel

    class Action(BaseModel):
        metadata: dict = Field(default_factory=dict)

    class Observation(BaseModel):
        done: bool = False
        reward: Optional[float] = None
        metadata: dict = Field(default_factory=dict)


class KMCAction(Action):
    action_type: str = Field(..., description="allocate_resources | resolve_conflict | enforce_rule | adapt_policy | investigate | self_restrain | noop")
    params: Dict[str, Any] = Field(default_factory=dict)


class KMCObservation(Observation):
    message: str = Field("")
    episode_id: str = Field("")
    step_count: int = Field(0)
    stakeholders: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    resources: Dict[str, float] = Field(default_factory=dict)
    active_conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    rules: List[str] = Field(default_factory=list)
    alerts: List[str] = Field(default_factory=list)
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)