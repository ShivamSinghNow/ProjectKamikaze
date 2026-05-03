"""Control plane: drone roster + (later) AIP bridge.

KAM-5 ships /health, POST /register, POST /heartbeat, GET /roster, and a
periodic roster log with liveness-based eviction (drones silent for more
than HEARTBEAT_TIMEOUT_S seconds get dropped). KAM-12 will extend this
with Foundry/AIP streaming.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from kamikaze_common.logging import get_logger
from kamikaze_common.schemas import Heartbeat, RegisterReq, RegisterResp, RosterResp

log = get_logger("control-plane")

# drone_id -> {hostname, registered_at, last_heartbeat}
ROSTER: dict[str, dict] = {}

HEARTBEAT_TIMEOUT_S = 10.0  # evict if no heartbeat for this long
EVICTION_INTERVAL_S = 2.0
ROSTER_LOG_INTERVAL_S = 5.0


def _evict_stale(now: float) -> list[str]:
    evicted: list[str] = []
    for drone_id, meta in list(ROSTER.items()):
        if now - meta["last_heartbeat"] > HEARTBEAT_TIMEOUT_S:
            ROSTER.pop(drone_id, None)
            evicted.append(drone_id)
    return evicted


async def _liveness_evictor() -> None:
    while True:
        await asyncio.sleep(EVICTION_INTERVAL_S)
        gone = _evict_stale(time.time())
        for drone_id in gone:
            log.warning("evicted %s (no heartbeat for >%.1fs)", drone_id, HEARTBEAT_TIMEOUT_S)


async def _roster_logger() -> None:
    while True:
        await asyncio.sleep(ROSTER_LOG_INTERVAL_S)
        ids = sorted(ROSTER.keys())
        log.info("roster tick | count=%d | drones=%s", len(ids), ids)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("control-plane up — waiting for drones")
    tasks = [
        asyncio.create_task(_roster_logger()),
        asyncio.create_task(_liveness_evictor()),
    ]
    try:
        yield
    finally:
        for t in tasks:
            t.cancel()


app = FastAPI(title="kamikaze-control-plane", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/register")
async def register(req: RegisterReq) -> RegisterResp:
    now = time.time()
    ROSTER[req.drone_id] = {
        "hostname": req.hostname,
        "registered_at": now,
        "last_heartbeat": now,
    }
    log.info("registered %s (host=%s) | roster=%d", req.drone_id, req.hostname, len(ROSTER))
    peers = [d for d in sorted(ROSTER.keys()) if d != req.drone_id]
    return RegisterResp(ok=True, peers=peers)


@app.post("/heartbeat")
async def heartbeat(hb: Heartbeat) -> dict[str, bool]:
    meta = ROSTER.get(hb.drone_id)
    if meta is None:
        # Unknown drone: ignore. The drone-agent will re-register on next tick
        # if it gets a 200 response with `known=False`.
        return {"ok": True, "known": False}
    meta["last_heartbeat"] = time.time()
    return {"ok": True, "known": True}


@app.get("/roster")
async def roster() -> RosterResp:
    ids = sorted(ROSTER.keys())
    return RosterResp(drones=ids, count=len(ids))
