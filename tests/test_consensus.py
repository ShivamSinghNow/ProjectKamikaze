from __future__ import annotations

import pytest
from drone_agent.consensus import ConsensusFusion
from kamikaze_common.schemas import BBox, DetectionEvent


def detection_event(
    drone_id: str,
    conf: float,
    *,
    t: float = 0.0,
    class_id: int = 1,
    world_pos: tuple[float, float, float] | None = None,
) -> DetectionEvent:
    return DetectionEvent(
        drone_id=drone_id,
        t=t,
        class_id=class_id,
        conf=conf,
        bbox=BBox(x=10.0, y=20.0, w=30.0, h=40.0),
        world_pos=world_pos,
    )


def test_single_detection_below_threshold_does_not_elect() -> None:
    fusion = ConsensusFusion()

    fusion.observe(detection_event("drone-1", 0.8))

    assert fusion.best_track(now=0.1) is None


def test_multiple_unique_drones_fuse_above_threshold() -> None:
    fusion = ConsensusFusion()

    fusion.observe(detection_event("drone-1", 0.7))
    fusion.observe(detection_event("drone-2", 0.7))

    track = fusion.best_track(now=0.1)

    assert track is not None
    assert track.id == "class:1"
    assert track.fused_conf == pytest.approx(0.91)
    assert track.seen_by == ["drone-1", "drone-2"]
    assert track.interceptor_id == "drone-1"


def test_duplicate_detection_from_same_drone_does_not_double_count() -> None:
    fusion = ConsensusFusion(conf_threshold=0.0)

    fusion.observe(detection_event("drone-1", 0.6, t=0.0))
    fusion.observe(detection_event("drone-1", 0.6, t=0.1))

    track = fusion.best_track(now=0.2)

    assert track is not None
    assert track.fused_conf == pytest.approx(0.6)
    assert track.seen_by == ["drone-1"]


def test_stale_detections_expire_after_ttl() -> None:
    fusion = ConsensusFusion(ttl_s=1.0)

    fusion.observe(detection_event("drone-1", 0.7, t=0.0))
    fusion.observe(detection_event("drone-2", 0.7, t=0.0))

    assert fusion.best_track(now=0.5) is not None
    assert fusion.best_track(now=1.01) is None


def test_world_pos_election_picks_nearest_live_drone_and_tie_breaks() -> None:
    fusion = ConsensusFusion(conf_threshold=0.0)

    fusion.observe(detection_event("drone-1", 0.4, world_pos=(0.0, 0.0, 0.0)))

    track = fusion.best_track(
        now=0.1,
        drone_positions={
            "drone-1": (5.0, 0.0, 0.0),
            "drone-2": (1.0, 0.0, 0.0),
            "drone-3": (1.0, 0.0, 0.0),
        },
    )

    assert track is not None
    assert track.world_pos == (0.0, 0.0, 0.0)
    assert track.interceptor_id == "drone-2"


def test_no_world_pos_fallback_picks_highest_confidence_contributor() -> None:
    fusion = ConsensusFusion(conf_threshold=0.0)

    fusion.observe(detection_event("drone-2", 0.8))
    fusion.observe(detection_event("drone-1", 0.8))
    fusion.observe(detection_event("drone-3", 0.7))

    track = fusion.best_track(now=0.1)

    assert track is not None
    assert track.world_pos is None
    assert track.interceptor_id == "drone-1"
