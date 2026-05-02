"""Policy interface. Frozen by KAM-5 — KAM-9 only swaps the implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from kamikaze_common.schemas import Track


@dataclass
class Observation:
    """What the policy sees each tick. KAM-9 may extend, never narrow."""

    drone_state: np.ndarray  # 12-vec: pos(3) + rpy(3) + vel(3) + ang_vel(3)
    track: Track | None  # fused detection from gossip mesh, if any


# RPMs for the four motors of a quadcopter.
MotorCmds = np.ndarray  # shape (4,)


class Policy(Protocol):
    def act(self, obs: Observation, dt: float) -> MotorCmds: ...
    def reset(self) -> None: ...
