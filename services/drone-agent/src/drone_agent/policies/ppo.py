"""PPOPolicy — KAM-9 primary. Loads a stable-baselines3 zip and wraps
`.predict(obs)` behind the Policy Protocol.

KAM-5 ships this as a stub: if stable-baselines3 isn't installed OR the
checkpoint doesn't exist, the policy registry catches the exception in
__init__ and falls back to NoOpPolicy.

CONTRACT FOR KAM-9: replace `act()` to assemble the real obs vector from
drone_state + obs.track per InterceptAviary's observation layout, and to
translate the model's velocity-setpoint action through DSLPIDControl into
4 motor RPMs. The placeholder below intentionally hovers (not crashes) on
shape mismatches so the demo doesn't crash-loop if KAM-9 ships a checkpoint
without updating this method.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from drone_agent.policies.noop import _HOVER_RPM
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
        self._warned = False

    def act(self, obs: Observation, dt: float) -> MotorCmds:  # noqa: ARG002
        # Guard: until KAM-9 wires obs/action into InterceptAviary's actual
        # shapes, predict() and reshape() will both raise on shape mismatch.
        # Crash-looping here with `restart: unless-stopped` would kill the
        # demo. Hover instead, and warn ONCE so the operator notices.
        try:
            action, _ = self._model.predict(obs.drone_state, deterministic=True)
            return np.asarray(action, dtype=np.float32).reshape(4)
        except Exception as exc:
            if not self._warned:
                from kamikaze_common.logging import get_logger

                get_logger("ppo").warning(
                    "PPOPolicy.act failed (%s) — hovering until KAM-9 wires "
                    "obs/action shapes. ckpt=%s",
                    exc,
                    self._ckpt,
                )
                self._warned = True
            return np.full((4,), _HOVER_RPM, dtype=np.float32)

    def reset(self) -> None:
        self._warned = False
