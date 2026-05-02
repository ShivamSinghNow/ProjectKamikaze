"""SwarmAviary — 5-drone cardinal-sentry formation env.

Subclass of CtrlAviary (control-only; no RL hooks). KAM-9's InterceptAviary
will subclass BaseRLAviary separately and reuse the same INITIAL_XYZS so
the training env and live runtime share the formation geometry.

Drone roles & geometry (chosen for omnidirectional Shahed detection):
  drone-1 (perimeter, N)  — (+R, 0, alt_p), camera faces +X
  drone-2 (perimeter, E)  — (0, -R, alt_p), camera faces -Y
  drone-3 (perimeter, S)  — (-R, 0, alt_p), camera faces -X
  drone-4 (perimeter, W)  — (0, +R, alt_p), camera faces +Y
  drone-5 (spotter)       — (0, 0, alt_s),  360° pano (4 views, KAM-10)
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Lazy import — kamikaze_common is imported by the drone-agent which does NOT
# install pybullet. The drone-agent never instantiates SwarmAviary; it just
# reads the constants below.
try:
    from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary  # type: ignore
    from gym_pybullet_drones.utils.enums import DroneModel, Physics  # type: ignore

    _HAVE_PYBULLET_DRONES = True
except Exception:  # pragma: no cover — drone-agent path
    CtrlAviary = object  # type: ignore[misc, assignment]
    DroneModel = None  # type: ignore[assignment]
    Physics = None  # type: ignore[assignment]
    _HAVE_PYBULLET_DRONES = False


NUM_DRONES = 5
DRONE_IDS = [f"drone-{i}" for i in range(1, NUM_DRONES + 1)]

PERIMETER_RADIUS = 2.0   # meters from origin
PERIMETER_ALT = 1.5
SPOTTER_ALT = 5.0

INITIAL_XYZS = np.array(
    [
        [+PERIMETER_RADIUS, 0.0,                 PERIMETER_ALT],   # drone-1 N
        [0.0,               -PERIMETER_RADIUS,   PERIMETER_ALT],   # drone-2 E
        [-PERIMETER_RADIUS, 0.0,                 PERIMETER_ALT],   # drone-3 S
        [0.0,               +PERIMETER_RADIUS,   PERIMETER_ALT],   # drone-4 W
        [0.0,               0.0,                 SPOTTER_ALT],     # drone-5 spotter
    ],
    dtype=np.float64,
)

# Yaw each perimeter drone outward so its camera faces away from the centroid.
# drone-1 faces +X (yaw 0), drone-2 faces -Y (yaw -90°), drone-3 faces -X (180°),
# drone-4 faces +Y (90°). Spotter yaw 0 — KAM-10 captures 4 views regardless.
INITIAL_RPYS = np.array(
    [
        [0.0, 0.0,  0.0],
        [0.0, 0.0, -np.pi / 2],
        [0.0, 0.0,  np.pi],
        [0.0, 0.0,  np.pi / 2],
        [0.0, 0.0,  0.0],
    ],
    dtype=np.float64,
)

ROLES = {
    "drone-1": "perimeter",
    "drone-2": "perimeter",
    "drone-3": "perimeter",
    "drone-4": "perimeter",
    "drone-5": "spotter",
}


class SwarmAviary(CtrlAviary):  # type: ignore[misc, valid-type]
    """5-drone cardinal-sentry env. Pass `gui=True` for the host spike,
    `gui=False` (default) for the headless sim service."""

    def __init__(
        self,
        gui: bool = False,
        ctrl_freq: int = 240,
        pyb_freq: int = 240,
        record: bool = False,
        **kwargs: Any,
    ) -> None:
        if not _HAVE_PYBULLET_DRONES:
            raise ImportError(
                "SwarmAviary requires gym-pybullet-drones. Install via "
                "`uv pip install -e packages/kamikaze_common[envs]` or "
                "use the kamikaze-common[envs] extra."
            )
        # CtrlAviary does NOT accept vision_attributes (that's BaseAviary-only).
        # KAM-10 will capture per-drone camera frames via env._getDroneImages(i)
        # with a manually-set IMG_RES — no constructor flag needed.
        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=NUM_DRONES,
            initial_xyzs=INITIAL_XYZS.copy(),
            initial_rpys=INITIAL_RPYS.copy(),
            physics=Physics.PYB,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obstacles=False,
            user_debug_gui=False,
            **kwargs,
        )
