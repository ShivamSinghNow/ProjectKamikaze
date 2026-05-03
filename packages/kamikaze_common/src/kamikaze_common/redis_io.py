"""Redis pub/sub helpers for the sim ↔ drone-agent dataplane and gossip mesh.

Channels:
  world:state         — sim publishes the full world snapshot at SIM_TICK_HZ.
                        msgpack: {"tick": int, "t": float,
                                  "drones": [{"id", "pos", "rpy",
                                              "vel", "ang_vel"}, ...]}
                        Camera frames land here in KAM-10 (skipped in KAM-8).
  drone:{id}:cmd      — drone-agent publishes motor RPMs each tick.
                        msgpack: {"rpms": [r1, r2, r3, r4]}
  detections:gossip   — drone-agent publishes compact KAM-11 detection gossip.
                        msgpack: {"d", "t", "c", "f", "b", optional "w"}
  tracks:fused        — drone-agent publishes fused KAM-11 consensus tracks.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import msgpack
import numpy as np
import redis

from kamikaze_common.schemas import BBox, DetectionEvent, Track

WORLD_STATE_CHANNEL = "world:state"
DETECTIONS_GOSSIP_CHANNEL = "detections:gossip"
TRACKS_FUSED_CHANNEL = "tracks:fused"


def cmd_channel(drone_id: str) -> str:
    return f"drone:{drone_id}:cmd"


def pack(payload: dict) -> bytes:
    return msgpack.packb(payload, use_bin_type=True)


def unpack(data: bytes) -> dict:
    return msgpack.unpackb(data, raw=False)


def detection_event_payload(event: DetectionEvent) -> dict:
    """Build the compact KAM-11 gossip shape.

    Short keys keep the msgpack payload small:
      d=drone_id, t=timestamp, c=class_id, f=confidence,
      b=[x,y,w,h], w=[x,y,z] world position if known.
    """

    payload = {
        "d": event.drone_id,
        "t": float(event.t),
        "c": int(event.class_id),
        "f": float(event.conf),
        "b": [
            float(event.bbox.x),
            float(event.bbox.y),
            float(event.bbox.w),
            float(event.bbox.h),
        ],
    }
    if event.world_pos is not None:
        payload["w"] = [float(v) for v in event.world_pos]
    return payload


def detection_event_from_payload(payload: dict) -> DetectionEvent:
    bbox = payload["b"]
    world_pos = payload.get("w")
    return DetectionEvent(
        drone_id=str(payload["d"]),
        t=float(payload["t"]),
        class_id=int(payload["c"]),
        conf=float(payload["f"]),
        bbox=BBox(
            x=float(bbox[0]),
            y=float(bbox[1]),
            w=float(bbox[2]),
            h=float(bbox[3]),
        ),
        world_pos=tuple(float(v) for v in world_pos) if world_pos is not None else None,
    )


def pack_detection_event(event: DetectionEvent) -> bytes:
    return pack(detection_event_payload(event))


def unpack_detection_event(data: bytes) -> DetectionEvent:
    return detection_event_from_payload(unpack(data))


def publish_detection_gossip(client: redis.Redis, event: DetectionEvent) -> int:
    return client.publish(DETECTIONS_GOSSIP_CHANNEL, pack_detection_event(event))


def track_payload(track: Track) -> dict:
    return track.model_dump(mode="json")


def pack_track(track: Track) -> bytes:
    return pack(track_payload(track))


def unpack_track(data: bytes) -> Track:
    return Track.model_validate(unpack(data))


def publish_fused_track(client: redis.Redis, track: Track) -> int:
    return client.publish(TRACKS_FUSED_CHANNEL, pack_track(track))


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
