"""FastAPI app for Hypernoa Astrum (OpenEnv compatible).

Provides HTTP endpoints for the Astrum adaptive RL environment.
Supports both the OpenEnv protocol and a standalone FastAPI server.
"""

from __future__ import annotations

from typing import Optional

try:
    from openenv.core.env_server import create_app as _openenv_create_app

    from .env import AstrumEnvironment
    from .models import AstrumAction, AstrumObservation

    def _create_env():
        return AstrumEnvironment()

    app = _openenv_create_app(
        _create_env,
        AstrumAction,
        AstrumObservation,
        env_name="hypernoa_astrum",
    )

except ImportError:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    from .env import AstrumEnvironment
    from .models import AstrumAction

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

    _env = AstrumEnvironment()

    class ResetRequest(BaseModel):
        seed: Optional[int] = None
        episode_id: Optional[str] = None

    @app.get("/health")
    def health():
        return {"status": "healthy", "env": "hypernoa_astrum", "version": "0.1.0"}

    @app.get("/")
    def root():
        return {
            "env": "hypernoa_astrum",
            "version": "0.1.0",
            "description": "Adaptive environment for aligned intelligence",
            "endpoints": {
                "GET /health": "Health check",
                "POST /reset": "Reset environment (optional: seed, episode_id)",
                "POST /step": "Take an action (action_type + params)",
            },
        }

    @app.post("/reset")
    def reset(req: ResetRequest | None = None):
        seed = req.seed if req else None
        episode_id = req.episode_id if req else None
        obs = _env.reset(seed=seed, episode_id=episode_id)
        return obs.model_dump()

    @app.post("/step")
    def step(action: AstrumAction):
        if _env._state is None:
            raise HTTPException(
                status_code=400,
                detail="Environment not initialized. Call /reset first.",
            )
        obs = _env.step(action)
        return obs.model_dump()


def run():
    """Entry point for `astrum-server` CLI command."""
    import uvicorn

    uvicorn.run(
        "hypernoa.astrum_env.server:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )
