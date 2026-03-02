from __future__ import annotations

import json
import logging
import time
from typing import Any
from urllib import error, request

logger = logging.getLogger(__name__)


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class LiveRenderServer:
    """Publish live metadata/frames to central FastAPI routes."""

    def __init__(
        self,
        *,
        output_path: str,
        game: str,
        host: str = "127.0.0.1",
        port: int = 8770,
        title: str = "Live",
        frame_stride: int = 1,
    ) -> None:
        self.output_path = output_path
        self.game = game
        self.host = host
        self.port = int(port)
        self.title = title
        self.frame_stride = max(1, int(frame_stride))

        self._frame_count = 0
        self._running = False
        self._next_retry_at = 0.0
        self._next_log_at = 0.0

    def _endpoint(self, path: str) -> str:
        return f"{self.url()}/{self.game}/live/{path.lstrip('/')}"

    def _post_json(self, path: str, payload: dict[str, Any]) -> bool:
        now = time.monotonic()
        if now < self._next_retry_at:
            return False

        try:
            body = json.dumps(payload, default=_json_default).encode("utf-8")
            req = request.Request(
                self._endpoint(path),
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=0.35):
                pass
            self._next_retry_at = 0.0
            return True
        except (error.URLError, TimeoutError, OSError, ValueError) as exc:
            # Avoid spamming logs inside hot frame loops.
            if now >= self._next_log_at:
                logger.warning("Live render publish failed: %s", exc)
                self._next_log_at = now + 5.0
            self._next_retry_at = now + 0.5
            return False

    def start(self) -> None:
        self._running = True
        self._frame_count = 0
        self.push_meta({"title": self.title, "game": self.game})

    def stop(self) -> None:
        self._running = False

    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def push_meta(self, meta: dict[str, Any]) -> None:
        if not self._running:
            return
        self._post_json("meta", {"metadata": meta})

    def push_frame(self, frame: dict[str, Any]) -> None:
        if not self._running:
            return
        self._frame_count += 1
        if self._frame_count % self.frame_stride != 0:
            return
        self._post_json("frame", {"frame": frame})


__all__ = ["LiveRenderServer"]
