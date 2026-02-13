from __future__ import annotations

import sys
from pathlib import Path

try:
    from server.snake.live_server import LiveRenderServer
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from server.snake.live_server import LiveRenderServer

__all__ = ["LiveRenderServer"]
