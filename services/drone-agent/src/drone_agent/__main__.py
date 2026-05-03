"""Drone-agent boot loop.

Per tick:
    state = sim_client.get_state()
    frame = sim_client.get_frame()       # KAM-10 fills with real RGB
    det   = perception_stub.detect(frame)
    mesh.publish_detection(det)          # when det is present
    POST /heartbeat -> control-plane (re-registers if evicted)
    cmds  = policy.act(obs, dt)          # obs.track only on elected interceptor
    sim_client.publish_cmd(cmds)         # sim drives physics with these RPMs
"""

from __future__ import annotations

import sys
import time

import httpx
import numpy as np
from kamikaze_common.logging import get_logger
from kamikaze_common.schemas import Heartbeat, RegisterReq, RegisterResp

from drone_agent import config as cfg_mod
from drone_agent import perception_stub
from drone_agent.policies import build as build_policy
from drone_agent.policy import Observation
from drone_agent.redis_mesh import RedisGossipMesh
from drone_agent.sim_client import SimClient


def _register(cfg: cfg_mod.DroneConfig, client: httpx.Client, log) -> list[str]:
    req = RegisterReq(drone_id=cfg.drone_id, hostname=cfg.drone_id)
    last_err: Exception | None = None
    for attempt in range(30):
        try:
            r = client.post("/register", json=req.model_dump(), timeout=2.0)
            r.raise_for_status()
            resp = RegisterResp.model_validate(r.json())
            log.info("POST /register -> %d ok | peers=%s", r.status_code, resp.peers)
            return resp.peers
        except Exception as exc:
            last_err = exc
            log.info("control-plane not ready yet (attempt %d): %s", attempt + 1, exc)
            time.sleep(1.0)
    raise RuntimeError(f"could not register with control-plane: {last_err}")


_HB_FORCE_REREGISTER_AFTER = 5  # consecutive failures
_HB_LOG_EVERY_N = 30  # one WARN per ~30 ticks during outage instead of every tick


class _HeartbeatState:
    """Bookkeeping for circuit-breaker behavior on /heartbeat outages."""

    def __init__(self) -> None:
        self.fail_count = 0


def _send_heartbeat(
    client: httpx.Client, hb: Heartbeat, log, state: _HeartbeatState
) -> bool:
    """Returns False when we should re-register (CP evicted us OR we've
    missed too many heartbeats in a row to trust our own roster entry)."""
    try:
        r = client.post("/heartbeat", json=hb.model_dump(), timeout=2.0)
        r.raise_for_status()
        if state.fail_count > 0:
            log.info("heartbeat recovered after %d failures", state.fail_count)
        state.fail_count = 0
        return bool(r.json().get("known", True))
    except Exception as exc:
        state.fail_count += 1
        # Throttle the warn flood: log on first failure, then every 30 ticks.
        if state.fail_count == 1 or state.fail_count % _HB_LOG_EVERY_N == 0:
            log.warning(
                "heartbeat post failed (%d in a row): %s",
                state.fail_count,
                exc,
            )
        # After N misses, force a re-register attempt so we self-heal even
        # if the control-plane comes back up clean and never returns known=False.
        return state.fail_count < _HB_FORCE_REREGISTER_AFTER


def main() -> int:
    cfg = cfg_mod.load()
    log = get_logger("drone", drone_id=cfg.drone_id)

    log.info(
        "booted | role=%s policy=%s | tick=%.1fHz | fusion=%.2f ttl=%.2fs",
        cfg.role,
        cfg.policy,
        cfg.tick_hz,
        cfg.fusion_conf_threshold,
        cfg.fusion_ttl_s,
    )

    policy = build_policy(cfg)
    actual_cls = policy.__class__.__name__
    expected_cls = {
        "noop": "NoOpPolicy",
        "ppo": "PPOPolicy",
        "pronav": "ProNavMlpPolicy",
    }.get(cfg.policy)
    if expected_cls is not None and actual_cls != expected_cls:
        log.warning(
            "POLICY=%s requested but running %s (FALLBACK)", cfg.policy, actual_cls
        )
    else:
        log.info("policy=%s ready (%s)", cfg.policy, actual_cls)

    sim_client = SimClient(cfg.redis_url, cfg.drone_id)
    sim_client.start()
    mesh = RedisGossipMesh(
        cfg.redis_url,
        cfg.drone_id,
        ttl_s=cfg.fusion_ttl_s,
        conf_threshold=cfg.fusion_conf_threshold,
        log=log,
    )
    mesh.start()

    cp_client = httpx.Client(base_url=cfg.control_plane_url)
    _register(cfg, cp_client, log)
    hb_state = _HeartbeatState()

    # Don't block waiting for the first world:state — heartbeats must keep
    # flowing or control-plane evicts us at 10s. The loop tolerates state=None.
    dt = 1.0 / cfg.tick_hz
    log_every_n = max(1, int(cfg.tick_hz))  # ~1 log line/sec
    t_start = time.time()
    tick = 0
    try:
        while True:
            state = sim_client.get_state()
            mesh.set_drone_positions(sim_client.get_drone_positions())
            frame = sim_client.get_frame()
            det = perception_stub.detect(frame)
            now = time.time()
            t = now - t_start

            hb = Heartbeat(drone_id=cfg.drone_id, t=t, detection=det)
            if det is not None:
                world_pos = getattr(det, "world_pos", None)
                mesh.publish_detection(det, t=now, world_pos=world_pos)

            if not _send_heartbeat(cp_client, hb, log, hb_state):
                reason = (
                    f"{hb_state.fail_count} consecutive heartbeat failures"
                    if hb_state.fail_count >= _HB_FORCE_REREGISTER_AFTER
                    else "control-plane evicted us"
                )
                log.warning("re-registering (%s)", reason)
                _register(cfg, cp_client, log)
                hb_state.fail_count = 0

            if state is not None:
                drone_state = np.concatenate(
                    [state.pos, state.rpy, state.vel, state.ang_vel]
                ).astype(np.float32)
            else:
                drone_state = np.zeros(12, dtype=np.float32)

            track = mesh.track_for_policy(cfg.drone_id, now=now)
            obs = Observation(drone_state=drone_state, track=track)
            cmds = policy.act(obs, dt)
            sim_client.publish_cmd(cmds)

            if tick % log_every_n == 0:
                pos_str = f"[{state.pos[0]:.2f}, {state.pos[1]:.2f}, {state.pos[2]:.2f}]" if state else "?"
                track_id = track.id if track is not None else None
                log.info(
                    "tick=%d t=%.1fs | pos=%s | det=%s | policy_track=%s | sim_rx=%d",
                    tick,
                    t,
                    pos_str,
                    det,
                    track_id,
                    sim_client.payload_count,
                )

            tick += 1
            time.sleep(dt)
    finally:
        mesh.stop()
        sim_client.stop()
        cp_client.close()


if __name__ == "__main__":
    sys.exit(main() or 0)
