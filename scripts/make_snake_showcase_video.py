#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pygame

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simverse.envs.snake.render import (  # noqa: E402
    SIDEBAR_WIDTH,
    _draw_obs_frame,
    _extract_reward_value,
    _extract_scalar_int,
    _infer_length_from_obs,
    _infer_pos_from_obs,
    _termination_label,
)


def _episode_number(path: Path) -> int | None:
    match = re.search(r"episode_(\d+)\.json$", path.name)
    if match is None:
        return None
    return int(match.group(1))


def _selected_files(recordings_dir: Path, episode_ranges: Iterable[tuple[int, int]]) -> list[Path]:
    files = sorted(recordings_dir.glob("episode_*.json"))
    selected: list[Path] = []
    for path in files:
        ep = _episode_number(path)
        if ep is None:
            continue
        for start, end in episode_ranges:
            if start <= ep <= end:
                selected.append(path)
                break
    return selected


def _load_frames(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    frames = data.get("frames", [])
    return frames if isinstance(frames, list) else []


def _infer_grid_shape(paths: list[Path]) -> tuple[int, int]:
    for path in paths:
        for frame in _load_frames(path):
            obs = np.asarray(frame.get("observation"), dtype=np.float32)
            if obs.ndim == 3 and obs.shape[0] >= 4:
                return int(obs.shape[2]), int(obs.shape[1])
    raise RuntimeError("No valid snake observations found in selected episode files")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create snake showcase video from replay JSON files",
    )
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=Path("recordings/snake"),
        help="Directory containing episode_*.json files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("recordings/snake/snake_showcase_ep001-010_ep100-110_normal.mp4"),
        help="Output mp4 path",
    )
    parser.add_argument("--fps", type=int, default=18, help="Output FPS")
    parser.add_argument("--cell-size", type=int, default=28, help="Render cell size")
    parser.add_argument("--crf", type=int, default=20, help="libx264 CRF")
    parser.add_argument(
        "--episodes",
        type=str,
        default="1-10,100-110",
        help="Comma-separated episode ranges, e.g. 1-10,100-110",
    )
    return parser.parse_args()


def _parse_episode_ranges(spec: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" not in token:
            value = int(token)
            ranges.append((value, value))
            continue
        left, right = token.split("-", 1)
        start = int(left.strip())
        end = int(right.strip())
        if end < start:
            start, end = end, start
        ranges.append((start, end))
    if not ranges:
        raise ValueError("No valid episode ranges provided")
    return ranges


def main() -> None:
    args = parse_args()
    episode_ranges = _parse_episode_ranges(args.episodes)
    selected = _selected_files(args.recordings_dir, episode_ranges)
    if not selected:
        raise SystemExit(f"No episode JSON files matched ranges in {args.recordings_dir}")

    grid_w, grid_h = _infer_grid_shape(selected)
    width_px = grid_w * int(args.cell_size) + SIDEBAR_WIDTH
    height_px = grid_h * int(args.cell_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    screen = pygame.display.set_mode((width_px, height_px))
    font = pygame.font.SysFont("Verdana", 18)
    title_font = pygame.font.SysFont("Verdana", 22, bold=True)

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width_px}x{height_px}",
        "-r",
        str(max(1, int(args.fps))),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(int(args.crf)),
        str(args.output),
    ]
    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if proc.stdin is None:
        raise RuntimeError("Failed to open ffmpeg stdin")

    frame_count = 0
    used_files = 0
    try:
        for replay_path in selected:
            frames = _load_frames(replay_path)
            if not frames:
                continue

            replay_file_used = False
            replay_ep = _episode_number(replay_path) or 0
            for frame in frames:
                obs = np.asarray(frame.get("observation"), dtype=np.float32)
                if obs.ndim != 3 or obs.shape[0] < 4:
                    continue
                if obs.shape[2] != grid_w or obs.shape[1] != grid_h:
                    continue

                info = frame.get("info", {}) if isinstance(frame.get("info", {}), dict) else {}
                step = _extract_scalar_int(frame.get("step"), default=0)
                episode = _extract_scalar_int(frame.get("episode"), default=replay_ep)
                score = _extract_scalar_int(info.get("score"), default=0)
                inferred_length = _infer_length_from_obs(obs)
                length = _extract_scalar_int(
                    info.get("snake_length", info.get("slength")),
                    default=inferred_length,
                )
                length = max(length, inferred_length)
                term_reason = _extract_scalar_int(info.get("termination_reason"), default=0)
                reward = _extract_reward_value(frame.get("rewards"))
                done = bool(frame.get("done", False))
                status = "done" if done else "running"

                head_pos = info.get("head_pos")
                if head_pos is None:
                    head_pos = _infer_pos_from_obs(obs[2])
                food_pos = info.get("food_pos")
                if food_pos is None:
                    food_pos = _infer_pos_from_obs(obs[1])

                metrics = [
                    ("Replay", replay_path.name),
                    ("Episode", str(episode)),
                    ("Step", str(step)),
                    ("State", status),
                    ("Termination", f"{_termination_label(term_reason)} ({term_reason})"),
                    ("Score", str(score)),
                    ("Length", str(length)),
                    ("Reward", f"{reward:.3f}"),
                    ("Head", str(head_pos)),
                    ("Food", str(food_pos)),
                    ("FPS", str(max(1, int(args.fps)))),
                ]
                _draw_obs_frame(
                    screen=screen,
                    font=font,
                    title_font=title_font,
                    cell_size=int(args.cell_size),
                    obs=obs,
                    panel_title="Snake Replay",
                    panel_metrics=metrics,
                    panel_footer=None,
                )
                rgb = pygame.surfarray.array3d(screen).swapaxes(0, 1)
                proc.stdin.write(rgb.tobytes())
                frame_count += 1
                replay_file_used = True

            if replay_file_used:
                used_files += 1
    finally:
        proc.stdin.close()
        ret = proc.wait()
        pygame.quit()
        if ret != 0:
            raise RuntimeError(f"ffmpeg encode failed with code {ret}")

    print(f"output={args.output}")
    print(f"files_selected={len(selected)} files_used={used_files} frames_written={frame_count}")
    print(f"episodes={args.episodes} fps={int(args.fps)}")


if __name__ == "__main__":
    main()
