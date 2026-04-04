from __future__ import annotations


import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import default_config
from Kmc_space.server.kmc_environment import KMCEnvironment
from models import KMCAction, KMCObservation

_openenv_available = False
try:
    import openenv.core
    _openenv_available = True
except ImportError:
    pass

app = FastAPI(
    title="KMC",
    description="KMC adaptive environment for training coordinated intelligence",
    version="0.1.0",
)

_cors_origins: list[str] = os.environ.get(
    "KMC_CORS_ORIGINS", "http://localhost:3000,http://localhost:7860"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_env = KMCEnvironment(config=default_config())
_env_lock = threading.Lock()


class ResetRequest(BaseModel):
    seed: Optional[int] = None
    episode_id: Optional[str] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]
    timeout_s: Optional[float] = None
    request_id: Optional[str] = None


def _assert_initialized() -> None:
    if _env._state is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call POST /reset first.",
        )


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/")
def root_info():
    return {
        "env": "kmc",
        "version": "0.1.0",
        "openenv": _openenv_available,
        "description": "KMC adaptive environment for training coordinated intelligence",
        "endpoints": {
            "GET /": "This page",
            "GET /health": "Health check",
            "POST /reset": "Reset environment (optional: seed, episode_id)",
            "POST /step": "Take an action (action_type + params)",
            "GET /state": "Current environment state",
            "GET /metadata": "Action/observation space description",
            "GET /schema": "JSON schemas for action and observation models",
        },
    }


@app.post("/reset")
def reset(req: Optional[ResetRequest] = Body(default=None)):
    with _env_lock:
        obs = _env.reset(
            seed=req.seed if req else None,
            episode_id=req.episode_id if req else None,
        )
    return {"observation": obs.model_dump(), "done": False}


@app.post("/step")
def step(req: StepRequest):
    with _env_lock:
        _assert_initialized()
        action = KMCAction(
            action_type=req.action.get("action_type", "noop"),
            params=req.action.get("params", {}),
        )
        obs = _env.step(action)
    return {
        "observation": obs.model_dump(),
        "done": obs.done,
        "reward": obs.reward,
        "request_id": req.request_id,
    }


@app.get("/state")
def get_state():
    with _env_lock:
        state = _env._state
    if state is None:
        return {"episode_id": None, "step_count": 0, "initialized": False}
    return {
        "episode_id": state.episode_id,
        "step_count": state.step_count,
        "initialized": True,
    }


@app.get("/metadata")
def metadata():
    return {
        "env_name": "kmc",
        "version": "0.1.0",
        "openenv_compatible": _openenv_available,
        "action_space": {
            "types": [
                "allocate_resources",
                "resolve_conflict",
                "enforce_rule",
                "adapt_policy",
                "investigate",
                "self_restrain",
                "noop",
            ],
            "notes": {
                "timeout_s": "Accepted but not yet implemented.",
                "request_id": "Echoed back in /step response for caller correlation.",
            },
        },
        "observation_space": {
            "fields": [
                "stakeholders",
                "resources",
                "active_conflicts",
                "rules",
                "alerts",
                "reward",
                "reward_breakdown",
            ]
        },
    }


@app.get("/schema")
def schema():
    return {
        "action": KMCAction.model_json_schema(),
        "observation": KMCObservation.model_json_schema(),
    }


def main():
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("KMC_PORT", 7860)),
    )


if __name__ == "__main__":
    main()