"""Sim stub. KAM-8 swaps in headless gym-pybullet-drones (`p.connect(p.DIRECT)`).

Interface frozen: `tick() -> Frame`. Returns a black 640x480 RGB image so
perception_stub has something to chew on without pulling pybullet at boot.
"""

from __future__ import annotations

import numpy as np

Frame = np.ndarray


def tick() -> Frame:
    return np.zeros((480, 640, 3), dtype=np.uint8)
