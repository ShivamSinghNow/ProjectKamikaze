"""Perception stub. KAM-7/10 swap in real YOLO26 ONNX inference here.

Interface frozen: `detect(frame) -> Optional[Detection]`. The drone-agent
main loop never imports onnxruntime directly.
"""

from __future__ import annotations

import numpy as np

from kamikaze_common.schemas import Detection


def detect(frame: np.ndarray) -> Detection | None:  # noqa: ARG001 — frame unused in stub
    return None
