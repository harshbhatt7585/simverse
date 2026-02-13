from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from server.live.stream import live_stream_registry

router = APIRouter(prefix="/snake", tags=["snake"])


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _replay_dir() -> Path:
    configured = os.getenv("SNAKE_REPLAY_DIR", "recordings/snake")
    return Path(configured)


def _all_replay_files() -> list[Path]:
    replay_dir = _replay_dir()
    if not replay_dir.exists():
        return []
    return sorted(p for p in replay_dir.glob("*.json") if p.is_file())


@router.get("/")
def snake_root() -> dict[str, str]:
    return {"service": "snake", "status": "ok"}


@router.get("/snapshot")
def get_snapshot() -> dict[str, Any]:
    files = _all_replay_files()
    live_snapshot = live_stream_registry.get("snake").snapshot()
    return {
        "replay_dir": str(_replay_dir()),
        "replay_count": len(files),
        "latest_replay": files[-1].name if files else None,
        "live": live_snapshot,
    }


@router.get("/events")
@router.get("/live/events")
def get_events() -> StreamingResponse:
    stream = live_stream_registry.get("snake")

    def event_stream():
        subscriber = stream.subscribe()
        try:
            while True:
                yield stream.next_event(subscriber)
        finally:
            stream.unsubscribe(subscriber)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/live/meta")
def post_live_meta(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        if isinstance(payload, dict):
            metadata = payload
        else:
            raise HTTPException(status_code=400, detail="metadata must be a JSON object")
    published = live_stream_registry.get("snake").publish_meta(metadata)
    return {"status": "ok", **published}


@router.post("/live/frame")
def post_live_frame(payload: dict[str, Any]) -> dict[str, Any]:
    frame = payload.get("frame")
    if not isinstance(frame, dict):
        if isinstance(payload, dict):
            frame = payload
        else:
            raise HTTPException(status_code=400, detail="frame must be a JSON object")
    published = live_stream_registry.get("snake").publish_frame(frame)
    return {"status": "ok", **published}


@router.get("/replays")
@router.get("/replays/")
def get_replays() -> dict[str, list[dict[str, str]]]:
    episodes: list[dict[str, str]] = []
    for file in _all_replay_files():
        episodes.append({"id": file.stem, "name": file.name})
    return {"episodes": episodes}


@router.get("/replays/{replay_id}")
def get_replay(replay_id: str) -> dict[str, Any]:
    candidates = [replay_id, f"{replay_id}.json"]
    replay_dir = _replay_dir()
    for name in candidates:
        path = replay_dir / name
        if path.exists() and path.is_file():
            return {"id": path.stem, "name": path.name, "data": _read_json(path)}
    raise HTTPException(status_code=404, detail=f"Replay not found: {replay_id}")
