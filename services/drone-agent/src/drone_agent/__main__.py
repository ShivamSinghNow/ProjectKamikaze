"""Drone-agent boot loop.

Per tick:
    state = sim_client.get_state()
    frame = sim_client.get_frame()       # KAM-10 fills with real RGB
    det   = perception_stub.detect(frame)
    hb    = Heartbeat(...)
    mesh_stub.broadcast(hb)              # KAM-11 swap target
    POST /heartbeat -> control-plane (re-registers if evicted)
    cmds  = policy.act(obs, dt)
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
from drone_agent import mesh_stub, perception_stub
from drone_agent.policies import build as build_policy
from drone_agent.policy import Observation
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


def _send_heartbeat(client: httpx.Client, hb: Heartbeat, log) -> bool:
    """Returns False if the control-plane evicted us (we need to re-register)."""
    try:
        r = client.post("/heartbeat", json=hb.model_dump(), timeout=2.0)
        r.raise_for_status()
        return bool(r.json().get("known", True))
    except Exception as exc:
        log.warning("heartbeat post failed: %s", exc)
        return True  # don't trigger re-register on transient network errors


def main() -> int:
    cfg = cfg_mod.load()
    log = get_logger("drone", drone_id=cfg.drone_id)

    log.info(
        "booted | role=%s policy=%s | tick=%.1fHz",
        cfg.role,
        cfg.policy,
        cfg.tick_hz,
    )

    policy = build_policy(cfg)
    log.info("policy=%s ready (%s)", cfg.policy, policy.__class__.__name__)

    sim_client = SimClient(cfg.redis_url, cfg.drone_id)
    sim_client.start()

    cp_client = httpx.Client(base_url=cfg.control_plane_url)
    _register(cfg, cp_client, log)

    # Don't block waiting for the first world:state — heartbeats must keep
    # flowing or control-plane evicts us at 10s. The loop tolerates state=None.
    dt = 1.0 / cfg.tick_hz
    log_every_n = max(1, int(cfg.tick_hz))  # ~1 log line/sec
    t_start = time.time()
    tick = 0
    while True:
        state = sim_client.get_state()
        frame = sim_client.get_frame()
        det = perception_stub.detect(frame)
        t = time.time() - t_start

        hb = Heartbeat(drone_id=cfg.drone_id, t=t, detection=det)
        mesh_stub.broadcast(hb)

        if not _send_heartbeat(cp_client, hb, log):
            log.warning("control-plane evicted us — re-registering")
            _register(cfg, cp_client, log)

        if state is not None:
            drone_state = np.concatenate(
                [state.pos, state.rpy, state.vel, state.ang_vel]
            ).astype(np.float32)
        else:
            drone_state = np.zeros(12, dtype=np.float32)

        obs = Observation(drone_state=drone_state, track=None)
        cmds = policy.act(obs, dt)
        sim_client.publish_cmd(cmds)

        if tick % log_every_n == 0:
            pos_str = f"[{state.pos[0]:.2f}, {state.pos[1]:.2f}, {state.pos[2]:.2f}]" if state else "?"
            log.info(
                "tick=%d t=%.1fs | pos=%s | det=%s | sim_rx=%d",
                tick,
                t,
                pos_str,
                det,
                sim_client.payload_count,
            )

        tick += 1
        time.sleep(dt)


if __name__ == "__main__":
    sys.exit(main() or 0)
