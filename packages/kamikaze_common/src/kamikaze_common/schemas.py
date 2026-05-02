"""Wire shapes shared between drone-agent, control-plane, and (later) the
Foundry/AIP HUD. Frozen now so KAM-10/11/12 don't have to renegotiate."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class Detection(BaseModel):
    bbox: BBox
    conf: float = Field(ge=0.0, le=1.0)
    class_id: int


class Track(BaseModel):
    """A fused detection consensus across the swarm. KAM-11 produces these."""

    id: str
    drone_id: str
    fused_conf: float = Field(ge=0.0, le=1.0)


class RegisterReq(BaseModel):
    drone_id: str
    hostname: str


class RegisterResp(BaseModel):
    ok: bool
    peers: list[str]


class Heartbeat(BaseModel):
    drone_id: str
    t: float
    detection: Detection | None = None


class RosterResp(BaseModel):
    drones: list[str]
    count: int
