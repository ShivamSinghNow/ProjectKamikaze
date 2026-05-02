"""PPOPolicy — KAM-9 primary. Loads a stable-baselines3 zip and wraps
`.predict(obs)` behind the Policy Protocol.

KAM-5 ships this as a stub: if stable-baselines3 isn't installed OR the
checkpoint doesn't exist, the policy registry catches the exception and
falls back to NoOpPolicy. The drone always boots.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from drone_agent.policy import MotorCmds, Observation


class PPOPolicy:
    def __init__(self, checkpoint_path: str) -> None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"PPO checkpoint not found at {checkpoint_path} — "
                "KAM-9 has not produced one yet"
            )

        # Lazy import — keeps NoOp boot cold-start fast and image lean.
        from stable_baselines3 import PPO  # type: ignore

        self._model: Any = PPO.load(checkpoint_path)
        self._ckpt = checkpoint_path

    def act(self, obs: Observation, dt: float) -> MotorCmds:  # noqa: ARG002
        # KAM-9: assemble model_obs from drone_state + track per InterceptAviary's
        # observation layout, then translate `action` (velocity setpoint) through
        # DSLPIDControl into 4-RPM motor cmds.
        action, _ = self._model.predict(obs.drone_state, deterministic=True)
        # Placeholder: pretend `action` is already RPMs. KAM-9 replaces this.
        return np.asarray(action, dtype=np.float32).reshape(4)

    def reset(self) -> None:
        pass
