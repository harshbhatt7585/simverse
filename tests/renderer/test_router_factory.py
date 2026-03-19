from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "renderer"))

from server.main import app  # noqa: E402


def test_snake_replay_route_accepts_direct_replay_file_path(tmp_path, monkeypatch) -> None:
    replay_file = tmp_path / "replay.json"
    replay_file.write_text(json.dumps({"frames": [{"step": 7}]}), encoding="utf-8")

    monkeypatch.setenv("SNAKE_REPLAY_DIR", str(tmp_path / "missing"))

    client = TestClient(app)

    response = client.get("/snake/replay/", params={"dir": str(replay_file)})
    assert response.status_code == 200
    assert response.json()["name"] == "replay.json"
    assert response.json()["data"]["frames"][0]["step"] == 7

    list_response = client.get("/snake/replays/", params={"dir": str(replay_file)})
    assert list_response.status_code == 200
    assert list_response.json()["episodes"] == [{"id": "replay", "name": "replay.json"}]
