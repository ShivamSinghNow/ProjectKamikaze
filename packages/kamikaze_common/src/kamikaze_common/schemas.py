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


class DetectionEvent(BaseModel):
    """Compact per-drone detection gossip payload."""

    drone_id: str
    t: float
    class_id: int
    conf: float = Field(ge=0.0, le=1.0)
    bbox: BBox
    world_pos: tuple[float, float, float] | None = None


class Track(BaseModel):
    """A fused detection consensus across the swarm. KAM-11 produces these.

    All consumers (KAM-9 PPO obs, KAM-11 closest-interceptor election,
    KAM-12 AIP HUD) need spatial data — this schema is the wire shape they
    all see, so we ship it complete now to avoid a mid-hackathon migration.
    """

    id: str
    drone_id: str  # which drone first / most-recently saw it
    fused_conf: float = Field(ge=0.0, le=1.0)
    bbox: BBox | None = None  # last bbox in the observer's image frame
    world_pos: tuple[float, float, float] | None = None  # for KAM-9 obs + interceptor election
    world_vel: tuple[float, float, float] | None = None  # closing-velocity reward (KAM-9)
    t: float  # last-seen wall-clock seconds since epoch
    seen_by: list[str] = Field(default_factory=list)  # drone IDs that contributed to fusion
    interceptor_id: str | None = None


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
