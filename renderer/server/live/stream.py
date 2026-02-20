from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


@dataclass(eq=False)
class _Subscriber:
    events: queue.Queue[str]


class LiveStream:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._meta: dict[str, Any] = {}
        self._last_frame: dict[str, Any] | None = None
        self._updated_at = 0.0
        self._subscribers: set[_Subscriber] = set()

    def _publish(self, event: str, payload: dict[str, Any]) -> None:
        message = _sse(event, payload)
        stale: list[_Subscriber] = []
        for subscriber in self._subscribers:
            try:
                subscriber.events.put_nowait(message)
            except queue.Full:
                stale.append(subscriber)
        for subscriber in stale:
            self._subscribers.discard(subscriber)

    def publish_meta(self, metadata: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            self._meta = metadata
            self._updated_at = time.time()
            payload = {"seq": self._seq, "metadata": self._meta}
            self._publish("meta", payload)
            return payload

    def publish_frame(self, frame: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            self._last_frame = frame
            self._updated_at = time.time()
            payload = {"seq": self._seq, "frame": self._last_frame}
            self._publish("frame", payload)
            return payload

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "seq": self._seq,
                "updated_at": self._updated_at,
                "metadata": self._meta,
                "last_frame": self._last_frame,
            }

    def subscribe(self) -> _Subscriber:
        subscriber = _Subscriber(events=queue.Queue(maxsize=256))
        with self._lock:
            self._subscribers.add(subscriber)
            snapshot = {
                "seq": self._seq,
                "updated_at": self._updated_at,
                "metadata": self._meta,
                "last_frame": self._last_frame,
            }
        try:
            subscriber.events.put_nowait(_sse("snapshot", snapshot))
        except queue.Full:
            pass
        return subscriber

    def unsubscribe(self, subscriber: _Subscriber) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)

    def next_event(self, subscriber: _Subscriber, timeout_sec: float = 15.0) -> str:
        try:
            return subscriber.events.get(timeout=timeout_sec)
        except queue.Empty:
            return ": ping\n\n"


class LiveStreamRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._streams: dict[str, LiveStream] = {}

    def get(self, game: str) -> LiveStream:
        normalized = game.strip().lower()
        with self._lock:
            stream = self._streams.get(normalized)
            if stream is None:
                stream = LiveStream()
                self._streams[normalized] = stream
            return stream


live_stream_registry = LiveStreamRegistry()
