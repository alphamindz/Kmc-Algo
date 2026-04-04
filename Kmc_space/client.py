"""OpenEnv client for the Astrum environment.

Connects to a running Astrum server (local or HF Space) and provides
sync/async access to reset() and step().

Usage:
    from astrum_env import AstrumEnv, AstrumAction

    with AstrumEnv(base_url="https://abnaidu-hypernoa-astrum.hf.space").sync() as env:
        result = env.reset()
        result = env.step(AstrumAction(
            action_type="allocate_resources",
            params={"stakeholder": "workers", "amount": 10, "resource": "budget"}
        ))
        print(result.reward)
"""

from openenv.core.env_client import EnvClient

from .models import AstrumAction, AstrumObservation


class AstrumEnv(EnvClient[AstrumAction, AstrumObservation]):
    """Client for the Hypernoa Astrum environment."""

    @property
    def action_type(self):
        return AstrumAction

    @property
    def observation_type(self):
        return AstrumObservation
