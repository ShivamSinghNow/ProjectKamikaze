"""NoOpPolicy — hover RPMs. KAM-5 default; lets compose come up green."""

from __future__ import annotations

import numpy as np

from drone_agent.policy import MotorCmds, Observation

# Hover RPM for the Crazyflie 2.x in gym-pybullet-drones (HB1 quadcopter).
# Real value lives in DSLPIDControl.HOVER_RPM; we hardcode it for KAM-5 so
# this module has zero external imports beyond numpy.
_HOVER_RPM = 14_468.0


class NoOpPolicy:
    def act(self, obs: Observation, dt: float) -> MotorCmds:  # noqa: ARG002
        return np.full((4,), _HOVER_RPM, dtype=np.float32)

    def reset(self) -> None:
        pass
