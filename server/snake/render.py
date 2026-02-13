from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

app = FastAPI()


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


@app.get("/")
def read_root():
    return {"service": "snake-render", "status": "ok"}


@app.get("/snapshot")
def get_snapshot():
    files = _all_replay_files()
    return {
        "replay_dir": str(_replay_dir()),
        "replay_count": len(files),
        "latest_replay": files[-1].name if files else None,
    }


@app.get("/events")
def get_events():
    return {"type": "noop", "message": "SSE stream not enabled yet"}


@app.get("/replays")
@app.get("/replays/")
@app.get("/snake/replays")
@app.get("/snake/replays/")
def get_replays():
    episodes: list[dict[str, Any]] = []
    for file in _all_replay_files():
        episodes.append({"id": file.stem, "name": file.name})
    return {"episodes": episodes}


@app.get("/replays/{replay_id}")
@app.get("/snake/replays/{replay_id}")
def get_replay(replay_id: str):
    candidates = [replay_id, f"{replay_id}.json"]
    replay_dir = _replay_dir()
    for name in candidates:
        path = replay_dir / name
        if path.exists() and path.is_file():
            return {"id": path.stem, "name": path.name, "data": _read_json(path)}
    raise HTTPException(status_code=404, detail=f"Replay not found: {replay_id}")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("SNAKE_RENDER_HOST", "127.0.0.1")
    port = int(os.getenv("SNAKE_RENDER_PORT", "8770"))
    uvicorn.run("server.snake.render:app", host=host, port=port, reload=False)
