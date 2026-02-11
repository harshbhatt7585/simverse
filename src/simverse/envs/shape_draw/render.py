from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

if __package__ is None or __package__.startswith("__main__"):
    _src = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_src))

import numpy as np
import pygame
import torch

from simverse.envs.shape_draw.config import ShapeDrawConfig
from simverse.envs.shape_draw.torch_env import ShapeDrawTorchEnv


@dataclass
class ReplayState:
    frames: List[dict[str, Any]] = field(default_factory=list)
    index: int = 0
    playing: bool = False
    source_path: str | None = None


class ShapeDrawRender:
    def __init__(self, size: int = 64, cell_size: int = 6, fps: int = 20) -> None:
        self.size = size
        self.cell_size = cell_size
        self.fps = fps

        pygame.init()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Menlo", 20)

        width = self.size * self.cell_size
        self.canvas_width = width
        self.panel_gap = 24
        self.screen = pygame.display.set_mode((width * 2 + self.panel_gap, width + 40))
        pygame.display.set_caption("ShapeDraw Renderer")

        self.replay = ReplayState()
        self.replay_exit_requested = False

    def close(self) -> None:
        pygame.quit()

    def _to_surface(self, array_2d: np.ndarray) -> pygame.Surface:
        img = (np.clip(array_2d, 0.0, 1.0) * 255.0).astype("uint8")
        rgb = img[:, :, None].repeat(3, axis=2)
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        return pygame.transform.scale(
            surf,
            (img.shape[1] * self.cell_size, img.shape[0] * self.cell_size),
        )

    def _draw_panels(self, target: np.ndarray, canvas: np.ndarray, hud: str) -> None:
        self.screen.fill((20, 22, 28))

        target_surface = self._to_surface(target)
        canvas_surface = self._to_surface(canvas)

        self.screen.blit(target_surface, (0, 40))
        self.screen.blit(canvas_surface, (self.canvas_width + self.panel_gap, 40))

        self.screen.blit(self.font.render("Target", True, (230, 230, 230)), (0, 8))
        self.screen.blit(
            self.font.render("Canvas", True, (230, 230, 230)),
            (self.canvas_width + self.panel_gap, 8),
        )
        self.screen.blit(
            self.font.render(hud, True, (150, 220, 140)),
            (10, self.canvas_width + 12),
        )

    def _extract_obs_channels(
        self,
        frame: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs = frame.get("observation")
        if obs is None:
            raise ValueError("Replay frame has no 'observation' field")
        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.ndim != 3 or obs_arr.shape[0] < 3:
            raise ValueError(f"Unexpected replay observation shape: {obs_arr.shape}")
        return obs_arr[0], obs_arr[1], obs_arr[2]

    def load_replay(self, replay_path: str) -> None:
        data = json.loads(Path(replay_path).read_text())
        self.load_replay_data(data, source_path=replay_path)

    def load_replay_data(self, data: dict[str, Any], source_path: str | None = None) -> None:
        frames = data.get("frames", [])
        if not frames:
            raise ValueError("Replay file contains no frames")

        first_canvas, _, _ = self._extract_obs_channels(frames[0])
        expected_shape = (self.size, self.size)
        if first_canvas.shape != expected_shape:
            raise ValueError(
                "Replay frame size does not match renderer size. "
                f"Replay={first_canvas.shape}, Renderer={expected_shape}"
            )

        self.replay.frames = frames
        self.replay.index = 0
        self.replay.playing = True
        self.replay.source_path = source_path
        self.replay_exit_requested = False

    def play_replay(self, loop: bool = False) -> None:
        if not self.replay.frames:
            raise ValueError("No replay loaded")

        paused = False

        while self.replay.playing:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.replay.playing = False
                    self.replay_exit_requested = True
                    raise SystemExit
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.replay.playing = False
                        self.replay_exit_requested = True
                        raise SystemExit
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    if event.key == pygame.K_LEFT:
                        self.replay.index = max(0, self.replay.index - 1)
                    if event.key == pygame.K_RIGHT:
                        self.replay.index = min(len(self.replay.frames) - 1, self.replay.index + 1)

            frame = self.replay.frames[self.replay.index]
            canvas, target, pen = self._extract_obs_channels(frame)
            step = int(frame.get("step", self.replay.index + 1))
            actions = frame.get("actions", [])
            action_value = "-"
            if isinstance(actions, list) and actions:
                first_action = actions[0]
                if isinstance(first_action, dict) and "action" in first_action:
                    action_value = str(first_action["action"])
            source_name = (
                Path(self.replay.source_path).name if self.replay.source_path else "replay"
            )
            pause_tag = " [paused]" if paused else ""
            hud = (
                f"{source_name} frame={self.replay.index + 1}/{len(self.replay.frames)} "
                f"step={step} action={action_value}{pause_tag}"
            )
            canvas_with_pen = np.clip(canvas + 0.5 * pen, 0.0, 1.0)
            self._draw_panels(target, canvas_with_pen, hud)
            pygame.display.flip()

            if not paused:
                self.replay.index += 1
                if self.replay.index >= len(self.replay.frames):
                    if loop:
                        self.replay.index = 0
                    else:
                        self.replay.playing = False

            self.clock.tick(self.fps)

    def wait_for_replay_quit(self) -> None:
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    waiting = False
            self.clock.tick(self.fps)

    def run_live(self) -> None:
        config = ShapeDrawConfig(
            width=self.size,
            height=self.size,
            num_agents=1,
            num_envs=1,
            max_steps=300,
        )
        env = ShapeDrawTorchEnv(config=config, num_envs=1, device="cpu")
        obs = env.reset()

        running = True
        while running:
            action = -1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        obs = env.reset()
                    elif event.key == pygame.K_UP:
                        action = ShapeDrawTorchEnv.ACTION_UP
                    elif event.key == pygame.K_DOWN:
                        action = ShapeDrawTorchEnv.ACTION_DOWN
                    elif event.key == pygame.K_LEFT:
                        action = ShapeDrawTorchEnv.ACTION_LEFT
                    elif event.key == pygame.K_RIGHT:
                        action = ShapeDrawTorchEnv.ACTION_RIGHT
                    elif event.key == pygame.K_SPACE:
                        action = ShapeDrawTorchEnv.ACTION_TOGGLE_PEN
                    elif event.key == pygame.K_q:
                        action = ShapeDrawTorchEnv.ACTION_BRUSH_DOWN
                    elif event.key == pygame.K_e:
                        action = ShapeDrawTorchEnv.ACTION_BRUSH_UP

            action_tensor = torch.tensor([[action]], dtype=torch.int64)
            obs, _reward, done, info = env.step(action_tensor)
            if bool(done[0].item()):
                obs = env.reset()

            obs_arr = obs["obs"][0].detach().cpu().numpy()
            canvas = obs_arr[0]
            target = obs_arr[1]
            similarity = float(info["similarity"][0].item())
            brush = int(info["brush"][0].item())
            pen_down = bool(info["pen_down"][0].item())

            hud = f"live sim={similarity:.3f} brush={brush} pen={'down' if pen_down else 'up'}"
            self._draw_panels(target, canvas, hud)
            pygame.display.flip()
            self.clock.tick(self.fps)


def _resolve_render_dir(args: argparse.Namespace) -> str | None:
    if args.replay_dir:
        return args.replay_dir
    if args.render_dir:
        return args.render_dir
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShapeDraw renderer")
    parser.add_argument("--size", type=int, default=64, help="Canvas width/height")
    parser.add_argument("--cell-size", type=int, default=6, help="Cell size in pixels")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second")
    parser.add_argument("--replay", type=str, default=None, help="Path to a replay JSON file")
    parser.add_argument("--replay-dir", type=str, default=None, help="Directory with replay JSON")
    parser.add_argument(
        "--render-dir",
        type=str,
        default=None,
        help="Alias for --replay-dir",
    )
    parser.add_argument("--loop", action="store_true", help="Loop replay when finished")
    args = parser.parse_args()

    replay_dir = _resolve_render_dir(args)

    renderer = ShapeDrawRender(size=args.size, cell_size=args.cell_size, fps=args.fps)
    try:
        if replay_dir:
            directory = Path(replay_dir)
            if not directory.exists():
                raise SystemExit(f"Replay directory not found: {directory}")
            replay_files = sorted(directory.glob("*.json"))
            if not replay_files:
                raise SystemExit(f"No replay JSON files found in {directory}")

            loop_all = True
            while loop_all:
                for replay_file in replay_files:
                    data = json.loads(replay_file.read_text())
                    renderer.load_replay_data(data, source_path=str(replay_file))
                    renderer.play_replay(loop=False)
                    if renderer.replay_exit_requested:
                        loop_all = False
                        break
                if not args.loop:
                    loop_all = False

            if not renderer.replay_exit_requested:
                renderer.wait_for_replay_quit()

        elif args.replay:
            renderer.load_replay(args.replay)
            renderer.play_replay(loop=args.loop)
            if not renderer.replay_exit_requested:
                renderer.wait_for_replay_quit()

        else:
            renderer.run_live()

    except SystemExit:
        pass
    finally:
        renderer.close()
