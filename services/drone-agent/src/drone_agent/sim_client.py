"""Redis client to the central sim service.

Replaces the KAM-5 sim_stub. The drone-agent never imports pybullet directly;
state arrives by subscribing to `world:state`, motor commands go out on
`drone:{id}:cmd`. KAM-10 will extend the world:state payload with camera
frames; until then `get_frame()` returns a black placeholder so
perception_stub keeps working unchanged.
"""

from __future__ import annotations

import threading
from contextlib import suppress
from dataclasses import dataclass

import numpy as np
import redis
from kamikaze_common.logging import get_logger
from kamikaze_common.redis_io import (
    WORLD_STATE_CHANNEL,
    publish_cmd,
    subscribe,
    unpack,
)

_FRAME_PLACEHOLDER = np.zeros((480, 640, 3), dtype=np.uint8)


@dataclass
class DroneState:
    pos: np.ndarray  # (3,)
    rpy: np.ndarray  # (3,)
    vel: np.ndarray  # (3,)
    ang_vel: np.ndarray  # (3,)


class SimClient:
    """Background world:state subscriber with synchronous reads."""

    def __init__(self, redis_url: str, drone_id: str) -> None:
        self._client = redis.Redis.from_url(redis_url)
        self._drone_id = drone_id
        self._latest_payload: dict | None = None
        self._payload_count = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pubsub = subscribe(self._client, [WORLD_STATE_CHANNEL])
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._log = get_logger("sim-client", drone_id=drone_id)

    def start(self) -> None:
        self._thread.start()
        self._log.info("subscribed to %s", WORLD_STATE_CHANNEL)

    def stop(self) -> None:
        self._stop.set()
        with suppress(Exception):
            self._pubsub.close()

    @property
    def payload_count(self) -> int:
        with self._lock:
            return self._payload_count

    def get_state(self) -> DroneState | None:
        """Return the latest pose for our drone, or None if no world:state
        has arrived yet (sim still spinning up)."""
        with self._lock:
            payload = self._latest_payload
        if payload is None:
            return None
        for d in payload["drones"]:
            if d["id"] == self._drone_id:
                return DroneState(
                    pos=np.asarray(d["pos"], dtype=np.float32),
                    rpy=np.asarray(d["rpy"], dtype=np.float32),
                    vel=np.asarray(d["vel"], dtype=np.float32),
                    ang_vel=np.asarray(d["ang_vel"], dtype=np.float32),
                )
        return None

    def get_drone_positions(self) -> dict[str, tuple[float, float, float]]:
        """Return latest live drone positions keyed by drone id."""
        with self._lock:
            payload = self._latest_payload
        if payload is None:
            return {}
        return {
            str(d["id"]): tuple(float(v) for v in d["pos"])
            for d in payload["drones"]
            if "id" in d and "pos" in d
        }

    def get_frame(self) -> np.ndarray:
        """Black placeholder until KAM-10 extends world:state with frames."""
        return _FRAME_PLACEHOLDER

    def publish_cmd(self, rpms: np.ndarray) -> int:
        return publish_cmd(self._client, self._drone_id, rpms)

    def _run(self) -> None:
        for msg in self._pubsub.listen():
            if self._stop.is_set():
                return
            data = msg.get("data")
            if not isinstance(data, (bytes, bytearray)):
                continue
            try:
                payload = unpack(bytes(data))
            except Exception as exc:
                self._log.warning("world:state unpack failed: %s", exc)
                continue
            with self._lock:
                self._latest_payload = payload
                self._payload_count += 1
