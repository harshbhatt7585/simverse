from __future__ import annotations

import json
import time
from pathlib import Path

from simverse.envs.snake.live_server import LiveRenderServer
from simverse.render_cli import build_render_parser


def _resolve_replay_sources(
    replay: str | None,
    replay_dir: str | None,
) -> tuple[list[Path], list[Path]]:
    replay_files: list[Path] = []
    replay_dirs: list[Path] = []

    if replay:
        replay_path = Path(replay)
        if not replay_path.exists():
            raise SystemExit(f"Replay path not found: {replay_path}")
        if replay_path.is_dir():
            replay_dirs.append(replay_path)
        else:
            replay_files.append(replay_path)

    if replay_dir:
        directory = Path(replay_dir)
        if not directory.exists():
            raise SystemExit(f"Replay directory not found: {directory}")
        if not directory.is_dir():
            raise SystemExit(f"Replay directory path is not a directory: {directory}")
        replay_dirs.append(directory)

    if not replay_files and not replay_dirs:
        raise SystemExit("Pass --replay <file_or_dir> and/or --replay-dir <dir>")

    for directory in replay_dirs:
        replay_files.extend(sorted(directory.glob("*.json")))

    replay_files = sorted(set(replay_files))
    if not replay_files:
        raise SystemExit("No replay JSON files found")
    return replay_files, replay_dirs


def parse_args():
    parser = build_render_parser(
        "Serve Snake replay in browser",
        [
            ("replay", {"help": "Replay JSON file or directory"}),
            ("replay_dir", {"help": "Directory containing replay JSON files"}),
            ("fps", {"default": 18}),
            ("loop", {"help": "Loop all replay files"}),
            ("watch", {"help": "Watch replay directory for new files"}),
            ("poll", {"default": 1.0, "help": "Watch poll interval"}),
            ("width", {"default": 15}),
            ("height", {"default": 15}),
        ],
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--output", type=str, default="recordings/snake/web_replay_live.jsonl")
    return parser.parse_args()


def _frame_dims(frame: dict, default_w: int, default_h: int) -> tuple[int, int]:
    obs = frame.get("observation")
    if isinstance(obs, list) and len(obs) >= 1 and isinstance(obs[0], list):
        h = len(obs[0])
        w = len(obs[0][0]) if h > 0 and isinstance(obs[0][0], list) else default_w
        return int(w), int(h)
    return default_w, default_h


def run(
    *,
    replay: str | None,
    replay_dir: str | None,
    fps: int,
    loop: bool,
    watch: bool,
    poll: float,
    host: str,
    port: int,
    output: str,
    width: int,
    height: int,
) -> None:
    replay_files, replay_dirs = _resolve_replay_sources(replay, replay_dir)

    server = LiveRenderServer(
        output_path=output,
        host=host,
        port=port,
        title="Snake Replay Web",
        frame_stride=1,
    )
    server.start()
    seen = set(replay_files)
    print(f"Open http://{host}:{port}")

    def _play_file(path: Path) -> None:
        data = json.loads(path.read_text())
        frames = data.get("frames", [])
        if not isinstance(frames, list) or not frames:
            return
        frame0 = frames[0]
        w, h = _frame_dims(frame0, width, height)
        server.push_meta(
            {
                "title": "Snake Replay Web",
                "env": "snake",
                "width": w,
                "height": h,
                "channels": 8,
            }
        )
        for frame in frames:
            server.push_frame(frame)
            time.sleep(1.0 / max(1, int(fps)))

    try:
        while True:
            for path in replay_files:
                _play_file(path)
            if watch and replay_dirs:
                watch_dir = replay_dirs[0]
                while True:
                    files = sorted(watch_dir.glob("*.json"))
                    new_files = [p for p in files if p not in seen]
                    if new_files:
                        replay_files.extend(new_files)
                        replay_files[:] = sorted(set(replay_files))
                        seen.update(new_files)
                        break
                    time.sleep(max(float(poll), 0.1))
                continue
            if not loop:
                break
    finally:
        server.stop()


if __name__ == "__main__":
    args = parse_args()
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
