#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from json import JSONDecodeError
from pathlib import Path

import numpy as np


def recover_frames(text: str) -> list[dict]:
    marker_idx = text.find('"frames"')
    if marker_idx < 0:
        return []
    array_start = text.find("[", marker_idx)
    if array_start < 0:
        return []

    decoder = json.JSONDecoder()
    frames: list[dict] = []
    pos = array_start + 1
    n = len(text)
    while pos < n:
        while pos < n and text[pos] in " \t\r\n,":
            pos += 1
        if pos >= n or text[pos] == "]":
            break
        try:
            obj, end = decoder.raw_decode(text, pos)
        except JSONDecodeError:
            break
        if isinstance(obj, dict):
            frames.append(obj)
        pos = end
    return frames


def load_frames(path: Path) -> tuple[list[dict], bool]:
    text = path.read_text()
    try:
        data = json.loads(text)
        return data.get("frames", []), False
    except JSONDecodeError:
        return recover_frames(text), True


def infer_obs_shape(files: list[Path]) -> tuple[int, int]:
    for path in files:
        frames, _ = load_frames(path)
        for frame in frames:
            obs = np.asarray(frame.get("observation"), dtype=np.float32)
            if obs.ndim == 3 and obs.shape[0] >= 2:
                return int(obs.shape[1]), int(obs.shape[2])
    raise RuntimeError("No valid observation frames found")


def generate_base_video(
    files: list[Path],
    out_path: Path,
    fps: int,
    scale: int,
    gap: int,
    crf: int,
) -> tuple[int, int, int]:
    height, width = infer_obs_shape(files)
    out_width = (width * 2 + gap) * scale
    out_height = height * scale

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{out_width}x{out_height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        str(out_path),
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if proc.stdin is None:
        raise RuntimeError("Failed to open ffmpeg stdin")

    total_frames = 0
    truncated_files = 0
    used_files = 0

    for path in files:
        frames, was_truncated = load_frames(path)
        if was_truncated:
            truncated_files += 1
        if not frames:
            continue

        wrote_this_file = 0
        for frame in frames:
            obs = np.asarray(frame.get("observation"), dtype=np.float32)
            if obs.ndim != 3 or obs.shape[0] < 2:
                continue
            if obs.shape[1] != height or obs.shape[2] != width:
                continue

            target = np.clip(obs[1], 0.0, 1.0)
            canvas = np.clip(obs[0], 0.0, 1.0)
            pen = np.clip(obs[2], 0.0, 1.0) if obs.shape[0] > 2 else np.zeros_like(canvas)
            canvas = np.clip(canvas + 0.5 * pen, 0.0, 1.0)

            left = (target * 255).astype(np.uint8)
            right = (canvas * 255).astype(np.uint8)
            rgb_left = np.repeat(left[:, :, None], 3, axis=2)
            rgb_right = np.repeat(right[:, :, None], 3, axis=2)

            combo = np.zeros((height, width * 2 + gap, 3), dtype=np.uint8)
            combo[:, :width] = rgb_left
            combo[:, width + gap :] = rgb_right

            scaled = np.repeat(np.repeat(combo, scale, axis=0), scale, axis=1)
            proc.stdin.write(scaled.tobytes())
            total_frames += 1
            wrote_this_file += 1

        if wrote_this_file > 0:
            used_files += 1

    proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg base encode failed with code {ret}")

    return total_frames, used_files, truncated_files


def generate_speed_video(base: Path, out: Path, speed: float, fps: int, crf: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(base),
            "-an",
            "-filter:v",
            f"setpts=PTS/{speed}",
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(crf),
            str(out),
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build shape_draw timelapse from recordings")
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=Path("recordings/shape_draw"),
        help="Directory containing episode_*.json recordings",
    )
    parser.add_argument(
        "--base-output",
        type=Path,
        default=Path("recordings/shape_draw/shape_draw_all_timelapse.mp4"),
        help="Output path for base timelapse",
    )
    parser.add_argument(
        "--speed-output",
        type=Path,
        default=Path("recordings/shape_draw/shape_draw_all_timelapse_8x.mp4"),
        help="Output path for sped-up timelapse",
    )
    parser.add_argument("--speed", type=float, default=8.0, help="Speed multiplier for fast video")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scale", type=int, default=6)
    parser.add_argument("--gap", type=int, default=8)
    parser.add_argument("--crf", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    files = sorted(args.recordings_dir.glob("episode_*.json"))
    if not files:
        raise SystemExit(f"No episode_*.json files found in {args.recordings_dir}")

    args.base_output.parent.mkdir(parents=True, exist_ok=True)
    args.speed_output.parent.mkdir(parents=True, exist_ok=True)

    total_frames, used_files, truncated_files = generate_base_video(
        files=files,
        out_path=args.base_output,
        fps=args.fps,
        scale=args.scale,
        gap=args.gap,
        crf=args.crf,
    )
    generate_speed_video(
        base=args.base_output,
        out=args.speed_output,
        speed=args.speed,
        fps=args.fps,
        crf=args.crf,
    )

    print(f"base_output={args.base_output}")
    print(f"speed_output={args.speed_output}")
    print(
        f"files_total={len(files)} files_used={used_files} "
        f"truncated_files={truncated_files} frames_written={total_frames}"
    )


if __name__ == "__main__":
    main()
