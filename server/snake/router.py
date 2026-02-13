from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

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
def get_snapshot() -> dict[str, str | int | None]:
    files = _all_replay_files()
    return {
        "replay_dir": str(_replay_dir()),
        "replay_count": len(files),
        "latest_replay": files[-1].name if files else None,
    }


@router.get("/events")
def get_events() -> dict[str, str]:
    return {"type": "noop", "message": "SSE stream not enabled yet"}


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
