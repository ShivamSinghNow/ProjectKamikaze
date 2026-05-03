from __future__ import annotations

from kamikaze_common.redis_io import pack_detection_event, unpack, unpack_detection_event
from kamikaze_common.schemas import BBox, DetectionEvent


def test_msgpack_detection_gossip_payload_round_trips() -> None:
    event = DetectionEvent(
        drone_id="drone-1",
        t=1_777_777_777.25,
        class_id=26,
        conf=0.83,
        bbox=BBox(x=1.0, y=2.0, w=3.0, h=4.0),
        world_pos=(5.0, 6.0, 7.0),
    )

    packed = pack_detection_event(event)
    raw_payload = unpack(packed)
    round_tripped = unpack_detection_event(packed)

    assert set(raw_payload) == {"d", "t", "c", "f", "b", "w"}
    assert len(packed) < 120
    assert round_tripped == event
