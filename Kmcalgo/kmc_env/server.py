"""FastAPI app for Kmcalgo KMC-Algo (OpenEnv compatible).

Provides HTTP endpoints for the KMC-Algo adaptive RL environment.
Supports both the OpenEnv protocol and a standalone FastAPI server.
"""

from __future__ import annotations
from typing import Optional

try:
    # Attempting to use OpenEnv's standard server creator
    from openenv.core.env_server import create_app as _openenv_create_app
    from .env import KmcalgoEnvironment
    from .models import KmcalgoAction, KmcalgoObservation

    def _create_env():
        return KmcalgoEnvironment()

    app = _openenv_create_app(
        _create_env,
        KmcalgoAction,
        KmcalgoObservation,
        env_name="kmc_algo",
    )

except ImportError:
    # Standalone FastAPI implementation if OpenEnv is not fully installed
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from .env import KmcalgoEnvironment
    from .models import KmcalgoAction

    app = FastAPI(
        title="Kmcalgo KMC-Algo",
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

    _env = KmcalgoEnvironment()

    class ResetRequest(BaseModel):
        seed: Optional[int] = None
        episode_id: Optional[str] = None

    @app.get("/health")
    def health():
        return {"status": "healthy", "env": "kmc_algo", "version": "0.1.0"}

    @app.get("/")
    def root():
        return {
            "env": "kmc_algo",
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
    def step(action: KmcalgoAction):
        # Using the private _state check from env.py logic
        if not hasattr(_env, '_state') or _env._state is None:
            raise HTTPException(
                status_code=400,
                detail="Environment not initialized. Call /reset first.",
            )
        obs = _env.step(action)
        return obs.model_dump()


def run():
    """Entry point for `kmc-server` CLI command."""
    import uvicorn
    # Make sure the import path matches your new folder structure
    uvicorn.run(
        "Kmcalgo.kmc_env.server:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )