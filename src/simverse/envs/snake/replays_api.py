from __future__ import annotations

import sys
from pathlib import Path

try:
    from server.snake.replays_api import SnakeReplayAPIServer, parse_args
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from server.snake.replays_api import SnakeReplayAPIServer, parse_args

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
