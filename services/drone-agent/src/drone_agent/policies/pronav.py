"""ProNavMlpPolicy — KAM-9 fallback. Closed-form Proportional Navigation
guidance + tiny MLP for engage/observe/abort. KAM-5 ships an empty stub."""

from __future__ import annotations

import numpy as np

from drone_agent.policies.noop import _HOVER_RPM
from drone_agent.policy import MotorCmds, Observation


class ProNavMlpPolicy:
    def act(self, obs: Observation, dt: float) -> MotorCmds:  # noqa: ARG002
        # KAM-9: implement ProNav (lambda_dot * N * V_closing), feed engage MLP,
        # convert to motor RPMs via DSLPIDControl.
        return np.full((4,), _HOVER_RPM, dtype=np.float32)

    def reset(self) -> None:
        pass
