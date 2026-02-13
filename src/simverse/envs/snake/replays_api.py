from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def _json_response(handler: BaseHTTPRequestHandler, code: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class SnakeReplayAPIServer:
    def __init__(
        self,
        replay_dir: str | Path,
        host: str = "127.0.0.1",
        port: int = 8770,
    ) -> None:
        self.replay_dir = Path(replay_dir)
        self.host = host
        self.port = int(port)
        self._server: ThreadingHTTPServer | None = None

    def _list_replay_files(self) -> list[Path]:
        if not self.replay_dir.exists():
            return []
        return sorted(path for path in self.replay_dir.glob("*.json") if path.is_file())

    def _load_replays(self) -> list[dict[str, Any]]:
        episodes: list[dict[str, Any]] = []
        for path in self._list_replay_files():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            episodes.append({"name": path.name, "data": payload})
        return episodes

    def start(self) -> None:
        server_state = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                return

            def do_GET(self) -> None:  # noqa: N802
                if self.path in {"/snake/replays", "/snake/replays/"}:
                    episodes = server_state._load_replays()
                    _json_response(self, 200, {"episodes": episodes})
                    return

                if self.path == "/healthz":
                    _json_response(self, 200, {"status": "ok"})
                    return

                _json_response(self, 404, {"error": "not found"})

        self.replay_dir.mkdir(parents=True, exist_ok=True)
        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        print(f"Snake Replay API: http://{self.host}:{self.port}/snake/replays/")
        print(f"Replay dir: {self.replay_dir}")
        self._server.serve_forever()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve Snake replay JSON files as a single API payload"
    )
    parser.add_argument(
        "--replay-dir",
        type=str,
        default="recordings/snake",
        help="Directory containing snake replay JSON files",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    server = SnakeReplayAPIServer(replay_dir=args.replay_dir, host=args.host, port=args.port)
    try:
        server.start()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
