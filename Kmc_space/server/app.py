"""FastAPI server for Hypernoa Astrum (OpenEnv compatible)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Optional

from models import AstrumAction, AstrumObservation
from server.astrum_environment import AstrumEnvironment
from config import default_config

_openenv_available = False
try:
    import openenv.core
    _openenv_available = True
except ImportError:
    pass

app = FastAPI(
    title="Hypernoa Astrum",
    description="Adaptive environment for training aligned intelligence",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_env = AstrumEnvironment(config=default_config())


class ResetRequest(BaseModel):
    seed: Optional[int] = None
    episode_id: Optional[str] = None


class StepRequest(BaseModel):
    action: dict[str, Any]
    timeout_s: Optional[float] = None
    request_id: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/")
def root_info():
    return {
        "env": "hypernoa_astrum",
        "version": "0.1.0",
        "openenv": _openenv_available,
        "description": "Adaptive environment for training aligned intelligence",
        "endpoints": {
            "GET /": "This page",
            "GET /health": "Health check",
            "POST /reset": "Reset environment (optional: seed, episode_id)",
            "POST /step": "Take an action (action_type + params)",
            "GET /state": "Get current environment state",
        },
    }


@app.post("/reset")
def reset(req: ResetRequest = None):
    global _env
    _env = AstrumEnvironment(config=default_config())
    seed = req.seed if req else None
    episode_id = req.episode_id if req else None
    obs = _env.reset(seed=seed, episode_id=episode_id)
    return {"observation": obs.model_dump(), "done": False}


@app.post("/step")
def step(req: StepRequest):
    if _env._state is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    action_data = req.action
    action = AstrumAction(
        action_type=action_data.get("action_type", "noop"),
        params=action_data.get("params", {}),
    )
    obs = _env.step(action)
    return {"observation": obs.model_dump(), "done": obs.done, "reward": obs.reward}


@app.get("/state")
def get_state():
    if _env._state is None:
        return {"episode_id": None, "step_count": 0, "initialized": False}
    return {
        "episode_id": _env._state.episode_id,
        "step_count": _env._state.step_count,
        "initialized": True,
    }


@app.get("/metadata")
def metadata():
    return {
        "env_name": "hypernoa_astrum",
        "version": "0.1.0",
        "openenv_compatible": _openenv_available,
        "action_space": {
            "types": ["allocate_resources", "resolve_conflict", "enforce_rule",
                       "adapt_policy", "investigate", "self_restrain", "noop"]
        },
        "observation_space": {
            "fields": ["stakeholders", "resources", "active_conflicts", "rules",
                        "alerts", "reward", "reward_breakdown"]
        },
    }


@app.get("/schema")
def schema():
    return {
        "action": AstrumAction.model_json_schema(),
        "observation": AstrumObservation.model_json_schema(),
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
