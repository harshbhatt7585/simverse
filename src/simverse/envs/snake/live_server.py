from __future__ import annotations

from typing import Any


class LiveRenderServer:
    """
    Minimal compatibility server.

    The FastAPI backend lives in `server/main.py`; this class keeps
    older training imports working without requiring the old HTTP/SSE server.
    """

    def __init__(
        self,
        output_path: str,
        host: str = "127.0.0.1",
        port: int = 8766,
        title: str = "Snake Live",
        frame_stride: int = 1,
    ) -> None:
        self.output_path = output_path
        self.host = host
        self.port = int(port)
        self.title = title
        self.frame_stride = int(frame_stride)

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def push_meta(self, meta: dict[str, Any]) -> None:
        return

    def push_frame(self, frame: dict[str, Any]) -> None:
        return


__all__ = ["LiveRenderServer"]
