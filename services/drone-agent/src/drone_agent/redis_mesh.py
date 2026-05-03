"""Redis-backed KAM-11 detection gossip mesh."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress

import redis
from kamikaze_common.redis_io import (
    DETECTIONS_GOSSIP_CHANNEL,
    TRACKS_FUSED_CHANNEL,
    publish_detection_gossip,
    publish_fused_track,
    subscribe,
    unpack_detection_event,
)
from kamikaze_common.schemas import Detection, DetectionEvent, Track

from drone_agent.consensus import FUSION_CONF_THRESHOLD, FUSION_TTL_S, ConsensusFusion


class RedisGossipMesh:
    """Background Redis pub/sub mesh plus local consensus view."""

    def __init__(
        self,
        redis_url: str,
        drone_id: str,
        ttl_s: float = FUSION_TTL_S,
        conf_threshold: float = FUSION_CONF_THRESHOLD,
        log=None,
    ) -> None:
        self._drone_id = drone_id
        self._pub_client = redis.Redis.from_url(redis_url)
        self._sub_client = redis.Redis.from_url(redis_url)
        self._pubsub = subscribe(self._sub_client, [DETECTIONS_GOSSIP_CHANNEL])
        self._fusion = ConsensusFusion(ttl_s=ttl_s, conf_threshold=conf_threshold)
        self._drone_positions: dict[str, tuple[float, float, float]] = {}
        self._latest_track: Track | None = None
        self._last_published_signature: tuple | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._log = log

    def start(self) -> None:
        self._thread.start()
        if self._log is not None:
            self._log.info(
                "redis gossip mesh subscribed | detections=%s fused=%s",
                DETECTIONS_GOSSIP_CHANNEL,
                TRACKS_FUSED_CHANNEL,
            )

    def stop(self) -> None:
        self._stop.set()
        with suppress(Exception):
            self._pubsub.close()

    def set_drone_positions(self, positions: Mapping[str, Sequence[float]]) -> None:
        live_positions: dict[str, tuple[float, float, float]] = {}
        for drone_id, position in positions.items():
            pos = tuple(float(value) for value in position)
            if len(pos) == 3:
                live_positions[drone_id] = pos
        with self._lock:
            self._drone_positions = live_positions

    def publish_detection(
        self,
        detection: Detection,
        t: float,
        world_pos: tuple[float, float, float] | None = None,
    ) -> None:
        event = DetectionEvent(
            drone_id=self._drone_id,
            t=t,
            class_id=detection.class_id,
            conf=detection.conf,
            bbox=detection.bbox,
            world_pos=world_pos,
        )
        try:
            publish_detection_gossip(self._pub_client, event)
        except Exception as exc:
            if self._log is not None:
                self._log.warning("detection gossip publish failed: %s", exc)
        self._ingest_event(event, now=t)

    def track_for_policy(self, drone_id: str, now: float | None = None) -> Track | None:
        track = self.current_track(now=now)
        if track is None or track.interceptor_id != drone_id:
            return None
        return track

    def current_track(self, now: float | None = None) -> Track | None:
        if now is None:
            now = time.time()
        with self._lock:
            self._latest_track = self._fusion.best_track(now, self._drone_positions)
            if self._latest_track is None:
                self._last_published_signature = None
            return self._latest_track

    def _run(self) -> None:
        for msg in self._pubsub.listen():
            if self._stop.is_set():
                return
            data = msg.get("data")
            if not isinstance(data, (bytes, bytearray)):
                continue
            try:
                event = unpack_detection_event(bytes(data))
            except Exception as exc:
                if self._log is not None:
                    self._log.warning("detection gossip unpack failed: %s", exc)
                continue
            if event.drone_id == self._drone_id:
                continue
            self._ingest_event(event, now=time.time())

    def _ingest_event(self, event: DetectionEvent, now: float) -> None:
        with self._lock:
            self._fusion.observe(event)
            track = self._fusion.best_track(now, self._drone_positions)
            self._latest_track = track
            if track is None:
                self._last_published_signature = None
            should_publish = track is not None and self._mark_published_locked(track)

        if track is not None and should_publish:
            try:
                subscribers = publish_fused_track(self._pub_client, track)
                if self._log is not None:
                    self._log.info(
                        "fused track | id=%s conf=%.3f seen_by=%s interceptor=%s subs=%d",
                        track.id,
                        track.fused_conf,
                        track.seen_by,
                        track.interceptor_id,
                        subscribers,
                    )
            except Exception as exc:
                if self._log is not None:
                    self._log.warning("fused track publish failed: %s", exc)

    def _mark_published_locked(self, track: Track) -> bool:
        signature = (
            track.id,
            track.interceptor_id,
            tuple(track.seen_by),
            round(track.fused_conf, 3),
            track.drone_id,
        )
        if signature == self._last_published_signature:
            return False
        self._last_published_signature = signature
        return True
