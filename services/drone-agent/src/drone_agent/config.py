"""Drone-agent runtime config — read once at boot from environment."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DroneConfig:
    drone_id: str
    role: str  # "perimeter" | "spotter"
    peers: list[str]
    control_plane_url: str
    redis_url: str
    policy: str  # "noop" | "ppo" | "pronav"
    ppo_checkpoint: str
    log_level: str
    tick_hz: float
    fusion_conf_threshold: float
    fusion_ttl_s: float


def load() -> DroneConfig:
    drone_id = os.environ.get("DRONE_ID", "drone-?")
    peers_raw = os.environ.get("PEERS", drone_id)
    # Exclude self — compose passes the same PEERS list to every drone, and
    # KAM-11's gossip mesh would otherwise broadcast to self / double-count
    # own detections during fusion.
    peers = [p.strip() for p in peers_raw.split(",") if p.strip() and p.strip() != drone_id]

    return DroneConfig(
        drone_id=drone_id,
        role=os.environ.get("ROLE", "perimeter").lower(),
        peers=peers,
        control_plane_url=os.environ.get("CONTROL_PLANE_URL", "http://control-plane:8000"),
        redis_url=os.environ.get("REDIS_URL", "redis://redis:6379"),
        policy=os.environ.get("POLICY", "noop").lower(),
        ppo_checkpoint=os.environ.get("PPO_CHECKPOINT", "/models/intercept_ppo.zip"),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        tick_hz=float(os.environ.get("TICK_HZ", "30.0")),
        fusion_conf_threshold=float(os.environ.get("FUSION_CONF_THRESHOLD", "0.9")),
        fusion_ttl_s=float(os.environ.get("FUSION_TTL_S", "1.0")),
    )
