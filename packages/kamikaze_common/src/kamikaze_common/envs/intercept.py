"""InterceptAviary stub — KAM-9 will fill in the reward function.

Subclass of gym-pybullet-drones' BaseRLAviary. Defined here (not in training/)
so both the trainer AND the live drone-agent can import the same obs/action
shapes without drift.
"""

from __future__ import annotations

from typing import Any

# Import lazily so a drone-agent running POLICY=noop never has to install
# gym-pybullet-drones. KAM-9 will replace this stub with the real subclass.
try:
    from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary  # type: ignore
    from gym_pybullet_drones.utils.enums import ActionType, ObservationType  # type: ignore

    _HAVE_PYBULLET_DRONES = True
except Exception:  # pragma: no cover — KAM-5 path
    BaseRLAviary = object  # type: ignore[misc, assignment]
    ActionType = None  # type: ignore[assignment]
    ObservationType = None  # type: ignore[assignment]
    _HAVE_PYBULLET_DRONES = False


class InterceptAviary(BaseRLAviary):  # type: ignore[misc, valid-type]
    """Single-drone interceptor env. KAM-9 implements the body."""

    def __init__(self, **kwargs: Any) -> None:
        if not _HAVE_PYBULLET_DRONES:
            raise ImportError(
                "InterceptAviary requires gym-pybullet-drones. "
                "Install with `uv sync` in the training/ workspace."
            )
        # KAM-9: pass obs=ObservationType.KIN, act=ActionType.VEL, set
        # episode length, spawn the Shahed target, etc.
        super().__init__(**kwargs)

    def _computeReward(self) -> float:  # KAM-9
        raise NotImplementedError("KAM-9: implement intercept reward")

    def _computeTerminated(self) -> bool:  # KAM-9
        raise NotImplementedError("KAM-9: implement termination (hit/miss)")

    def _computeTruncated(self) -> bool:  # KAM-9
        raise NotImplementedError("KAM-9: implement timeout truncation")

    def _computeInfo(self) -> dict[str, Any]:  # KAM-9
        return {}
