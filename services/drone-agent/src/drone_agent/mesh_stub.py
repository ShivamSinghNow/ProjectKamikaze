"""Mesh stub. KAM-11 swaps in Redis pub/sub or websockets gossip.

Interface frozen: `broadcast(msg) -> None`. The main loop just calls this
with a Heartbeat per tick.
"""

from __future__ import annotations

from kamikaze_common.logging import get_logger
from kamikaze_common.schemas import Heartbeat

_log = get_logger("mesh-stub")


def broadcast(msg: Heartbeat) -> None:
    _log.debug("mesh broadcast (stub) | %s", msg.model_dump_json())
