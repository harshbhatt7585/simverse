from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    from server.snake.web_render import parse_args, run
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from server.snake.web_render import parse_args, run

__all__ = ["parse_args", "run"]


if __name__ == "__main__":
    args: Any = parse_args()
    run(
        replay=args.replay,
        replay_dir=args.replay_dir,
        fps=args.fps,
        loop=args.loop,
        watch=args.watch,
        poll=args.poll,
        host=args.host,
        port=args.port,
        output=args.output,
        width=args.width,
        height=args.height,
    )
