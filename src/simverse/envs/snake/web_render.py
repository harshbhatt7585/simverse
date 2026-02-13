from __future__ import annotations

from simverse.envs.snake.replays_api import SnakeReplayAPIServer, parse_args

__all__ = ["SnakeWebRenderServer", "parse_args"]
SnakeWebRenderServer = SnakeReplayAPIServer


if __name__ == "__main__":
    args = parse_args()
    server = SnakeWebRenderServer(replay_dir=args.replay_dir, host=args.host, port=args.port)
    try:
        server.start()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
