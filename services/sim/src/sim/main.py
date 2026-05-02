"""Central sim service.

Loop at SIM_TICK_HZ:
  1. Drain any pending drone:{id}:cmd messages → update last_cmds
  2. env.step(last_cmds) — advance physics
  3. Build world:state payload from env.pos / env.rpy / env.vel / env.ang_v
  4. Publish on world:state
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import defaultdict

import numpy as np
import redis

from kamikaze_common.envs.swarm import DRONE_IDS, NUM_DRONES, SwarmAviary
from kamikaze_common.logging import get_logger
from kamikaze_common.redis_io import (
    cmd_channel,
    publish_world_state,
    state_payload,
    subscribe,
    unpack,
)

log = get_logger("sim")


class CmdListener:
    """Background thread that drains all drone:{id}:cmd channels into a
    {drone_id: rpms[4]} dict the main loop reads each tick."""

    def __init__(self, redis_url: str) -> None:
        self._client = redis.Redis.from_url(redis_url)
        self._pubsub = subscribe(self._client, [cmd_channel(d) for d in DRONE_IDS])
        self._latest: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._pubsub.close()
        except Exception:
            pass

    def latest(self) -> dict[str, np.ndarray]:
        with self._lock:
            return dict(self._latest)

    def cmd_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def _run(self) -> None:
        for msg in self._pubsub.listen():
            if self._stop.is_set():
                return
            channel = msg.get("channel")
            data = msg.get("data")
            if not channel or not isinstance(data, (bytes, bytearray)):
                continue
            try:
                ch = channel.decode() if isinstance(channel, bytes) else channel
                drone_id = ch.split(":", 2)[1]
                payload = unpack(bytes(data))
                rpms = np.asarray(payload["rpms"], dtype=np.float64)
                if rpms.shape != (4,):
                    log.warning("bad rpms shape from %s: %s", drone_id, rpms.shape)
                    continue
                with self._lock:
                    self._latest[drone_id] = rpms
                    self._counts[drone_id] += 1
            except Exception as exc:
                log.warning("cmd parse failed: %s", exc)


def main() -> int:
    redis_url = os.environ.get("REDIS_URL", "redis://redis:6379")
    tick_hz = float(os.environ.get("SIM_TICK_HZ", "30"))
    dt = 1.0 / tick_hz

    log.info("sim starting | redis=%s | tick=%.1fHz | drones=%s", redis_url, tick_hz, DRONE_IDS)

    env = SwarmAviary(gui=False, ctrl_freq=240, pyb_freq=240)
    env.reset(seed=0)
    log.info("SwarmAviary ready | num_drones=%d | initial pos=\n%s", NUM_DRONES, env.pos)

    pub_client = redis.Redis.from_url(redis_url)
    listener = CmdListener(redis_url)
    listener.start()

    # env.step expects motor RPMs in shape (num_drones, 4), one row per drone.
    last_cmds = np.full((NUM_DRONES, 4), env.HOVER_RPM, dtype=np.float64)
    log_every_n = int(tick_hz)  # ~1 log line/sec
    tick = 0
    t_start = time.time()

    try:
        while True:
            tick_start = time.time()
            cmds = listener.latest()
            for i, drone_id in enumerate(DRONE_IDS):
                if drone_id in cmds:
                    last_cmds[i] = cmds[drone_id]

            env.step(last_cmds)

            payload = state_payload(
                tick=tick,
                t=time.time() - t_start,
                drones=[
                    {
                        "id": DRONE_IDS[i],
                        "pos": env.pos[i],
                        "rpy": env.rpy[i],
                        "vel": env.vel[i],
                        "ang_vel": env.ang_v[i],
                    }
                    for i in range(NUM_DRONES)
                ],
            )
            n_subs = publish_world_state(pub_client, payload)

            if tick % log_every_n == 0:
                counts = listener.cmd_counts()
                log.info(
                    "tick=%d t=%.1fs subs=%d | cmd_counts=%s | drone-1 pos=%s",
                    tick,
                    payload["t"],
                    n_subs,
                    {k: counts.get(k, 0) for k in DRONE_IDS},
                    np.round(env.pos[0], 2).tolist(),
                )

            elapsed = time.time() - tick_start
            time.sleep(max(0.0, dt - elapsed))
            tick += 1
    except KeyboardInterrupt:
        log.info("sim shutting down")
        return 0
    finally:
        listener.stop()
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main() or 0)
