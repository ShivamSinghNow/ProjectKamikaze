"""Redis pub/sub helpers for the sim ↔ drone-agent dataplane.

Channels:
  world:state         — sim publishes the full world snapshot at SIM_TICK_HZ.
                        msgpack: {"tick": int, "t": float,
                                  "drones": [{"id", "pos", "rpy",
                                              "vel", "ang_vel"}, ...]}
                        Camera frames land here in KAM-10 (skipped in KAM-8).
  drone:{id}:cmd      — drone-agent publishes motor RPMs each tick.
                        msgpack: {"rpms": [r1, r2, r3, r4]}
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import msgpack
import numpy as np
import redis

WORLD_STATE_CHANNEL = "world:state"


def cmd_channel(drone_id: str) -> str:
    return f"drone:{drone_id}:cmd"


def pack(payload: dict) -> bytes:
    return msgpack.packb(payload, use_bin_type=True)


def unpack(data: bytes) -> dict:
    return msgpack.unpackb(data, raw=False)


def state_payload(tick: int, t: float, drones: list[dict]) -> dict:
    """Build the canonical world:state payload. `drones` items must have
    keys: id, pos (3,), rpy (3,), vel (3,), ang_vel (3,) — numpy arrays
    are accepted and converted to lists."""
    return {
        "tick": int(tick),
        "t": float(t),
        "drones": [_drone_dict(d) for d in drones],
    }


def _drone_dict(d: dict) -> dict:
    return {
        "id": d["id"],
        "pos": _to_list(d["pos"]),
        "rpy": _to_list(d["rpy"]),
        "vel": _to_list(d["vel"]),
        "ang_vel": _to_list(d["ang_vel"]),
    }


def _to_list(v: Any) -> list[float]:
    if isinstance(v, np.ndarray):
        return v.astype(float).tolist()
    return list(v)


def publish_world_state(client: redis.Redis, payload: dict) -> int:
    return client.publish(WORLD_STATE_CHANNEL, pack(payload))


def publish_cmd(client: redis.Redis, drone_id: str, rpms: np.ndarray | list[float]) -> int:
    return client.publish(cmd_channel(drone_id), pack({"rpms": _to_list(rpms)}))


def subscribe(client: redis.Redis, channels: Iterable[str]) -> redis.client.PubSub:
    """Subscribe to one or more channels. Returns a PubSub object the caller
    iterates with `.listen()` or polls with `.get_message()`."""
    pubsub = client.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(*channels)
    return pubsub
