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


def _selected_files_by_slice(
    recordings_dir: Path,
    *,
    start_episode: int,
    slice_size: int,
    skip_episodes: int,
) -> list[Path]:
    episode_to_file: dict[int, Path] = {}
    for path in sorted(recordings_dir.glob("episode_*.json")):
        ep = _episode_number(path)
        if ep is not None:
            episode_to_file[ep] = path
    if not episode_to_file:
        return []

    selected: list[Path] = []
    max_ep = max(episode_to_file)
    cursor = max(1, int(start_episode))
    slice_size = max(1, int(slice_size))
    skip_episodes = max(0, int(skip_episodes))
    step = slice_size + skip_episodes
    while cursor <= max_ep:
        for ep in range(cursor, cursor + slice_size):
            path = episode_to_file.get(ep)
            if path is not None:
                selected.append(path)
        cursor += step
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
        default=Path("recordings/snake/snake_showcase_slice2_skip100_ui.mp4"),
        help="Output mp4 path",
    )
    parser.add_argument("--fps", type=int, default=18, help="Output FPS")
    parser.add_argument("--cell-size", type=int, default=28, help="Render cell size")
    parser.add_argument("--crf", type=int, default=20, help="libx264 CRF")
    parser.add_argument(
        "--episodes",
        type=str,
        default="",
        help="Comma-separated episode ranges, e.g. 1-10,100-110",
    )
    parser.add_argument(
        "--slice-size",
        type=int,
        default=2,
        help="Episodes to include in each showcase slice when --episodes is not set",
    )
    parser.add_argument(
        "--skip-episodes",
        type=int,
        default=100,
        help="Episodes to skip between slices when --episodes is not set",
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=1,
        help="First episode index for slice mode when --episodes is not set",
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


def _draw_showcase_frame(
    *,
    screen: pygame.Surface,
    base_font: pygame.font.Font,
    title_font: pygame.font.Font,
    small_font: pygame.font.Font,
    cell_size: int,
    obs: np.ndarray,
    metrics: list[tuple[str, str]],
    progress_ratio: float,
) -> None:
    bg = (13, 18, 28)
    panel_bg = (20, 30, 47)
    panel_accent = (80, 169, 255)
    floor = (235, 241, 248)
    wall = (53, 65, 82)
    food = (231, 76, 60)
    body = (76, 201, 141)
    head = (35, 155, 86)
    label = (148, 163, 184)
    value = (242, 247, 255)

    h = int(obs.shape[1])
    w = int(obs.shape[2])
    grid_w_px = w * cell_size
    grid_h_px = h * cell_size
    screen.fill(bg)

    walls = obs[0]
    foods = obs[1] if obs.shape[0] > 1 else np.zeros_like(walls)
    heads = obs[2] if obs.shape[0] > 2 else np.zeros_like(walls)
    bodies = obs[3] if obs.shape[0] > 3 else np.zeros_like(walls)

    for y in range(h):
        for x in range(w):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, wall if walls[y, x] > 0.5 else floor, rect)

    for fy, fx in np.argwhere(foods > 0.5):
        rect = pygame.Rect(
            int(fx) * cell_size + 4,
            int(fy) * cell_size + 4,
            cell_size - 8,
            cell_size - 8,
        )
        pygame.draw.ellipse(screen, food, rect)

    for by, bx in np.argwhere(bodies > 0.5):
        rect = pygame.Rect(
            int(bx) * cell_size + 3,
            int(by) * cell_size + 3,
            cell_size - 6,
            cell_size - 6,
        )
        pygame.draw.rect(screen, body, rect, border_radius=max(3, cell_size // 5))

    for hy, hx in np.argwhere(heads > 0.5):
        rect = pygame.Rect(
            int(hx) * cell_size + 2,
            int(hy) * cell_size + 2,
            cell_size - 4,
            cell_size - 4,
        )
        pygame.draw.rect(screen, head, rect, border_radius=max(4, cell_size // 4))

    panel_rect = pygame.Rect(grid_w_px, 0, SIDEBAR_WIDTH, grid_h_px)
    pygame.draw.rect(screen, panel_bg, panel_rect)
    pygame.draw.line(screen, panel_accent, (grid_w_px, 0), (grid_w_px, grid_h_px), width=3)

    title = title_font.render("Snake Showcase", True, value)
    subtitle = small_font.render("UI refresh | replay montage", True, label)
    screen.blit(title, (grid_w_px + 16, 12))
    screen.blit(subtitle, (grid_w_px + 18, 40))

    bar_x = grid_w_px + 16
    bar_y = 66
    bar_w = SIDEBAR_WIDTH - 32
    bar_h = 12
    pygame.draw.rect(screen, (37, 52, 74), (bar_x, bar_y, bar_w, bar_h), border_radius=6)
    fill_w = int(max(0.0, min(1.0, progress_ratio)) * bar_w)
    if fill_w > 0:
        pygame.draw.rect(screen, panel_accent, (bar_x, bar_y, fill_w, bar_h), border_radius=6)

    y = 96
    for key, val in metrics:
        key_surf = base_font.render(key, True, label)
        val_surf = base_font.render(val, True, value)
        screen.blit(key_surf, (grid_w_px + 16, y))
        screen.blit(val_surf, (grid_w_px + 162, y))
        y += 24

    pygame.display.flip()


def main() -> None:
    args = parse_args()
    if args.episodes.strip():
        episode_ranges = _parse_episode_ranges(args.episodes)
        selected = _selected_files(args.recordings_dir, episode_ranges)
        selection_mode = f"ranges={args.episodes}"
    else:
        selected = _selected_files_by_slice(
            args.recordings_dir,
            start_episode=args.start_episode,
            slice_size=args.slice_size,
            skip_episodes=args.skip_episodes,
        )
        selection_mode = (
            f"slice={int(args.slice_size)} skip={int(args.skip_episodes)} "
            f"start={int(args.start_episode)}"
        )
    if not selected:
        raise SystemExit(f"No episode JSON files matched ranges in {args.recordings_dir}")

    grid_w, grid_h = _infer_grid_shape(selected)
    width_px = grid_w * int(args.cell_size) + SIDEBAR_WIDTH
    height_px = grid_h * int(args.cell_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    screen = pygame.display.set_mode((width_px, height_px))
    font = pygame.font.SysFont("Avenir Next", 18)
    title_font = pygame.font.SysFont("Avenir Next", 26, bold=True)
    small_font = pygame.font.SysFont("Avenir Next", 14)

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
    total_files = max(1, len(selected))
    try:
        for file_idx, replay_path in enumerate(selected):
            frames = _load_frames(replay_path)
            if not frames:
                continue

            replay_file_used = False
            replay_ep = _episode_number(replay_path) or 0
            total_frames_in_file = max(1, len(frames))
            for frame_idx, frame in enumerate(frames):
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

                progress_ratio = (file_idx + ((frame_idx + 1) / total_frames_in_file)) / total_files
                metrics = [
                    ("Replay", replay_path.name.replace(".json", "")),
                    ("Episode", str(episode)),
                    ("Step", str(step)),
                    ("Slice Mode", f"{int(args.slice_size)} on / {int(args.skip_episodes)} skip"),
                    ("State", status),
                    ("Termination", f"{_termination_label(term_reason)} ({term_reason})"),
                    ("Score", str(score)),
                    ("Length", str(length)),
                    ("Reward", f"{reward:.3f}"),
                    ("Head", str(head_pos)),
                    ("Food", str(food_pos)),
                    ("FPS", str(max(1, int(args.fps)))),
                ]
                _draw_showcase_frame(
                    screen=screen,
                    base_font=font,
                    title_font=title_font,
                    small_font=small_font,
                    cell_size=int(args.cell_size),
                    obs=obs,
                    metrics=metrics,
                    progress_ratio=progress_ratio,
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
    print(f"selection={selection_mode} fps={int(args.fps)}")


if __name__ == "__main__":
    main()
