"""Drone-agent boot loop.

Per tick:
    frame = sim_stub.tick()
    det   = perception_stub.detect(frame)
    hb    = Heartbeat(...)
    mesh_stub.broadcast(hb)        # KAM-11 swap target
    POST /heartbeat -> control-plane (re-registers if evicted)
    cmds  = policy.act(obs, dt)    # KAM-8 hands cmds back to the sim
"""

from __future__ import annotations

import sys
import time

import httpx
import numpy as np

from kamikaze_common.logging import get_logger
from kamikaze_common.schemas import Heartbeat, RegisterReq, RegisterResp

from drone_agent import config as cfg_mod
from drone_agent import mesh_stub, perception_stub, sim_stub
from drone_agent.policies import build as build_policy
from drone_agent.policy import Observation


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
        "booted | onnx=stub mesh=stub sim=stub policy=%s | tick=%.1fHz",
        cfg.policy,
        cfg.tick_hz,
    )

    policy = build_policy(cfg)
    log.info("policy=%s ready (%s)", cfg.policy, policy.__class__.__name__)

    client = httpx.Client(base_url=cfg.control_plane_url)
    _register(cfg, client, log)

    dt = 1.0 / cfg.tick_hz
    t_start = time.time()
    while True:
        frame = sim_stub.tick()
        det = perception_stub.detect(frame)
        t = time.time() - t_start

        hb = Heartbeat(drone_id=cfg.drone_id, t=t, detection=det)
        mesh_stub.broadcast(hb)

        if not _send_heartbeat(client, hb, log):
            log.warning("control-plane evicted us — re-registering")
            _register(cfg, client, log)

        obs = Observation(drone_state=np.zeros(12, dtype=np.float32), track=None)
        _cmds = policy.act(obs, dt)

        log.info("heartbeat tick (t=%.1fs) | det=%s", t, det)
        time.sleep(dt)


if __name__ == "__main__":
    sys.exit(main() or 0)
