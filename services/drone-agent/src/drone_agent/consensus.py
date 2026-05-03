"""Pure KAM-11 detection fusion and interceptor election."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from kamikaze_common.schemas import DetectionEvent, Track

FUSION_CONF_THRESHOLD = 0.9
FUSION_TTL_S = 1.0


def track_id_for(event: DetectionEvent) -> str:
    """KAM-11 v1 tracks one target bucket per detected class."""

    return f"class:{event.class_id}"


@dataclass
class ConsensusFusion:
    """Fuse the most recent detection from each drone within a short TTL."""

    ttl_s: float = FUSION_TTL_S
    conf_threshold: float = FUSION_CONF_THRESHOLD
    _detections: dict[tuple[str, str], DetectionEvent] = field(default_factory=dict)

    def observe(self, event: DetectionEvent) -> None:
        key = (track_id_for(event), event.drone_id)
        existing = self._detections.get(key)
        if existing is None or event.t >= existing.t:
            self._detections[key] = event

    def fused_tracks(
        self,
        now: float,
        drone_positions: Mapping[str, Sequence[float]] | None = None,
    ) -> list[Track]:
        self._purge_stale(now)
        grouped: dict[str, list[DetectionEvent]] = defaultdict(list)
        for (track_id, _drone_id), event in self._detections.items():
            grouped[track_id].append(event)

        tracks = [
            track
            for track_id, events in grouped.items()
            if (track := self._fuse_track(track_id, events, drone_positions)) is not None
        ]
        return sorted(tracks, key=lambda track: track.id)

    def best_track(
        self,
        now: float,
        drone_positions: Mapping[str, Sequence[float]] | None = None,
    ) -> Track | None:
        tracks = self.fused_tracks(now, drone_positions)
        if not tracks:
            return None
        return min(tracks, key=lambda track: (-track.fused_conf, track.id))

    def _purge_stale(self, now: float) -> None:
        stale_keys = [
            key for key, event in self._detections.items() if now - event.t > self.ttl_s
        ]
        for key in stale_keys:
            del self._detections[key]

    def _fuse_track(
        self,
        track_id: str,
        events: list[DetectionEvent],
        drone_positions: Mapping[str, Sequence[float]] | None,
    ) -> Track | None:
        miss_prob = 1.0
        for event in events:
            miss_prob *= 1.0 - event.conf
        fused_conf = 1.0 - miss_prob
        if fused_conf < self.conf_threshold:
            return None

        latest_event = min(events, key=lambda event: (-event.t, event.drone_id))
        world_event = min(
            (event for event in events if event.world_pos is not None),
            key=lambda event: (-event.t, event.drone_id),
            default=None,
        )
        world_pos = world_event.world_pos if world_event is not None else None
        interceptor_id = self._elect_interceptor(events, world_pos, drone_positions)

        return Track(
            id=track_id,
            drone_id=latest_event.drone_id,
            fused_conf=fused_conf,
            bbox=latest_event.bbox,
            world_pos=world_pos,
            t=latest_event.t,
            seen_by=sorted(event.drone_id for event in events),
            interceptor_id=interceptor_id,
        )

    def _elect_interceptor(
        self,
        events: list[DetectionEvent],
        world_pos: tuple[float, float, float] | None,
        drone_positions: Mapping[str, Sequence[float]] | None,
    ) -> str:
        if world_pos is not None and drone_positions:
            candidates: list[tuple[float, str]] = []
            for drone_id, position in drone_positions.items():
                pos = tuple(float(value) for value in position)
                if len(pos) != 3:
                    continue
                dist_sq = sum((pos[i] - world_pos[i]) ** 2 for i in range(3))
                candidates.append((dist_sq, drone_id))
            if candidates:
                return min(candidates, key=lambda item: (item[0], item[1]))[1]

        return min(events, key=lambda event: (-event.conf, event.drone_id)).drone_id
