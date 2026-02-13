from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import uvicorn
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve snake replay JSON files")
    parser.add_argument("--replay-dir", type=str, default="recordings/snake")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    return parser.parse_args()


class SnakeReplayAPIServer:
    def __init__(self, replay_dir: str, host: str = "127.0.0.1", port: int = 8770) -> None:
        self.replay_dir = replay_dir
        self.host = host
        self.port = int(port)

    def start(self) -> None:
        os.environ["SNAKE_REPLAY_DIR"] = self.replay_dir
        uvicorn.run("server.snake.render:app", host=self.host, port=self.port, reload=False)

    def stop(self) -> None:
        return


__all__ = ["SnakeReplayAPIServer", "parse_args"]


if __name__ == "__main__":
    args = parse_args()
    server = SnakeReplayAPIServer(replay_dir=args.replay_dir, host=args.host, port=args.port)
    try:
        server.start()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
