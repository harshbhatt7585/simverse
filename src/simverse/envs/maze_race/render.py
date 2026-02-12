from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __name__ == "__main__" and __package__ is None:
    _src = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_src))

import numpy as np
import pygame
import torch

from simverse.envs.maze_race.config import MazeRaceConfig
from simverse.envs.maze_race.torch_env import MazeRaceTorchEnv


class MazeRaceRenderer:
    def __init__(
        self,
        cell_size: int = 32,
        fps: int = 20,
        auto_mode: bool = True,
        auto_reset: bool = True,
    ) -> None:
        self.cell_size = cell_size
        self.fps = fps
        self.auto_mode = auto_mode
        self.auto_reset = auto_reset

        self.colors = {
            "bg": (18, 20, 24),
            "floor": (235, 239, 245),
            "wall": (53, 63, 81),
            "goal0": (64, 132, 255),
            "goal1": (255, 132, 84),
            "agent0": (22, 82, 214),
            "agent1": (214, 95, 30),
            "text": (245, 247, 250),
        }

    def run(self, env: MazeRaceTorchEnv) -> None:
        pygame.init()
        width = env.width * self.cell_size
        height = env.height * self.cell_size + 44
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Maze Race")
        font = pygame.font.SysFont("Verdana", 16)
        clock = pygame.time.Clock()

        env.reset()
        pending_actions = torch.zeros((1, env.num_agents), dtype=torch.int64, device=env.device)

        running = True
        while running:
            pending_actions.zero_()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        env.reset()
                    elif event.key == pygame.K_t:
                        self.auto_mode = not self.auto_mode

            if self.auto_mode:
                pending_actions.copy_(
                    torch.randint(
                        0,
                        5,
                        pending_actions.shape,
                        dtype=torch.int64,
                        device=env.device,
                    )
                )
            else:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]:
                    pending_actions[0, 0] = env.ACTION_UP
                elif keys[pygame.K_DOWN]:
                    pending_actions[0, 0] = env.ACTION_DOWN
                elif keys[pygame.K_LEFT]:
                    pending_actions[0, 0] = env.ACTION_LEFT
                elif keys[pygame.K_RIGHT]:
                    pending_actions[0, 0] = env.ACTION_RIGHT
                else:
                    pending_actions[0, 0] = env.ACTION_STAY

                if env.num_agents > 1:
                    if keys[pygame.K_w]:
                        pending_actions[0, 1] = env.ACTION_UP
                    elif keys[pygame.K_s]:
                        pending_actions[0, 1] = env.ACTION_DOWN
                    elif keys[pygame.K_a]:
                        pending_actions[0, 1] = env.ACTION_LEFT
                    elif keys[pygame.K_d]:
                        pending_actions[0, 1] = env.ACTION_RIGHT
                    else:
                        pending_actions[0, 1] = env.ACTION_STAY

            _obs, _reward, done, info = env.step(pending_actions)
            if bool(done[0].item()):
                if self.auto_mode and self.auto_reset:
                    env.reset()
                # Hold final frame until user resets.

            screen.fill(self.colors["bg"])

            walls = env.walls.detach().cpu().numpy()
            agent_positions = []
            for idx in range(env.num_agents):
                agent_positions.append(
                    (
                        int(env.agent_pos[0, idx, 0].item()),
                        int(env.agent_pos[0, idx, 1].item()),
                    )
                )

            for y in range(env.height):
                for x in range(env.width):
                    rect = pygame.Rect(
                        x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    )
                    color = self.colors["wall"] if walls[y, x] else self.colors["floor"]
                    pygame.draw.rect(screen, color, rect)

            goal_positions = getattr(env, "goal_positions", None)
            if not goal_positions:
                goal_positions = [env.goal0]
                if env.num_agents > 1:
                    goal_positions.append(env.goal1)
            goal_colors = [self.colors["goal0"], self.colors["goal1"]]
            for idx, (gx, gy) in enumerate(goal_positions):
                color = goal_colors[min(idx, len(goal_colors) - 1)]
                rect = pygame.Rect(
                    gx * self.cell_size,
                    gy * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(screen, color, rect)

            r = max(5, self.cell_size // 3)
            agent_colors = [self.colors["agent0"], self.colors["agent1"]]
            for idx, (px, py) in enumerate(agent_positions):
                color = agent_colors[min(idx, len(agent_colors) - 1)]
                pygame.draw.circle(
                    screen,
                    color,
                    (
                        px * self.cell_size + self.cell_size // 2,
                        py * self.cell_size + self.cell_size // 2,
                    ),
                    r,
                )

            winner = int(info["winner"][0].item())
            steps = int(info["steps"][0].item())
            if winner == 0:
                status = "winner: blue"
            elif winner == 1:
                status = "winner: orange"
            elif winner == env.WINNER_DRAW:
                status = "draw"
            else:
                status = "running"
            auto_status = "auto:on" if self.auto_mode else "auto:off"
            if env.num_agents > 1:
                controls = "arrows=blue, WASD=orange"
            else:
                controls = "arrows=agent"
            hud = f"steps={steps} | {status} | {auto_status} | {controls} | T=toggle auto, R=reset"
            text = font.render(hud, True, self.colors["text"])
            screen.blit(text, (8, env.height * self.cell_size + 12))

            pygame.display.flip()
            clock.tick(self.fps)

        pygame.quit()


class MazeRaceReplayRenderer:
    def __init__(self, cell_size: int = 32, fps: int = 20) -> None:
        self.cell_size = cell_size
        self.fps = fps
        self.colors = {
            "bg": (18, 20, 24),
            "floor": (235, 239, 245),
            "wall": (53, 63, 81),
            "goal0": (64, 132, 255),
            "goal1": (255, 132, 84),
            "agent0": (22, 82, 214),
            "agent1": (214, 95, 30),
            "text": (245, 247, 250),
        }
        self.screen: Optional[pygame.Surface] = None
        self.font: Optional[pygame.font.Font] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.running = True

    def _init_display(self, width: int, height: int) -> None:
        pygame.init()
        screen_width = width * self.cell_size
        screen_height = height * self.cell_size + 44
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Maze Race Replay")
        self.font = pygame.font.SysFont("Verdana", 16)
        self.clock = pygame.time.Clock()

    def _ensure_display(self, width: int, height: int) -> None:
        if self.screen is None:
            self._init_display(width, height)
            return
        current_width, current_height = self.screen.get_size()
        expected = (width * self.cell_size, height * self.cell_size + 44)
        if (current_width, current_height) != expected:
            self._init_display(width, height)

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False
                return False
        return True

    def _draw_frame(
        self,
        obs: np.ndarray,
        info: Dict[str, Any],
        step: int,
        done: bool,
    ) -> None:
        if self.screen is None or self.font is None or self.clock is None:
            return
        channel_count = obs.shape[0]
        if channel_count < 3:
            return
        num_agents = (channel_count - 1) // 2
        if 1 + 2 * num_agents != channel_count:
            return
        walls = obs[0]
        goal_maps = obs[1 : 1 + num_agents]
        agent_maps = obs[1 + num_agents : 1 + 2 * num_agents]

        height, width = walls.shape
        self._ensure_display(width, height)
        if self.screen is None:
            return

        self.screen.fill(self.colors["bg"])
        for y in range(height):
            for x in range(width):
                if walls[y, x] > 0.5:
                    color = self.colors["wall"]
                elif num_agents >= 1 and goal_maps[0][y, x] > 0.5:
                    color = self.colors["goal0"]
                elif num_agents >= 2 and goal_maps[1][y, x] > 0.5:
                    color = self.colors["goal1"]
                else:
                    color = self.colors["floor"]
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self.screen, color, rect)

        def _agent_pos(layer: np.ndarray) -> Tuple[int, int]:
            coords = np.argwhere(layer > 0.5)
            if coords.size == 0:
                return (0, 0)
            y, x = coords[0]
            return (int(x), int(y))

        r = max(5, self.cell_size // 3)
        agent_colors = [self.colors["agent0"], self.colors["agent1"]]
        for idx in range(num_agents):
            px, py = _agent_pos(agent_maps[idx])
            color = agent_colors[min(idx, len(agent_colors) - 1)]
            pygame.draw.circle(
                self.screen,
                color,
                (
                    px * self.cell_size + self.cell_size // 2,
                    py * self.cell_size + self.cell_size // 2,
                ),
                r,
            )

        winner = info.get("winner")
        if winner == 0:
            status = "winner: blue"
        elif winner == 1:
            status = "winner: orange"
        elif winner == -2:
            status = "draw"
        else:
            status = "running"
        step_info = info.get("steps", step)
        hud = f"steps={step_info} | {status} | replay"
        text = self.font.render(hud, True, self.colors["text"])
        self.screen.blit(text, (8, height * self.cell_size + 12))

        pygame.display.flip()
        self.clock.tick(self.fps)

        if done:
            self.clock.tick(self.fps)

    def play_frames(self, frames: List[Dict[str, Any]], loop: bool = False) -> None:
        if not frames:
            return
        idx = 0
        while self.running:
            if not self._handle_events():
                return
            frame = frames[idx]
            obs = np.asarray(frame.get("observation"))
            if obs.ndim != 3 or obs.shape[0] < 3:
                idx += 1
                if idx >= len(frames):
                    if loop:
                        idx = 0
                    else:
                        return
                continue
            info = frame.get("info", {})
            step = int(frame.get("step", idx + 1))
            done = bool(frame.get("done", False))
            self._draw_frame(obs, info, step, done)
            idx += 1
            if idx >= len(frames):
                if loop:
                    idx = 0
                else:
                    return

    def wait_screen(self, message: str) -> None:
        if self.screen is None:
            self._init_display(10, 4)
        if self.screen is None or self.font is None or self.clock is None:
            return
        self.screen.fill(self.colors["bg"])
        text = self.font.render(message, True, self.colors["text"])
        self.screen.blit(text, (12, 12))
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self) -> None:
        pygame.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render MazeRaceTorchEnv")
    parser.add_argument("--size", type=int, default=15, help="Maze width/height (square)")
    parser.add_argument("--cell", type=int, default=36, help="Cell size in pixels")
    parser.add_argument("--fps", type=int, default=20, help="Render FPS")
    parser.add_argument("--replay", type=str, default=None, help="Path to a replay JSON file")
    parser.add_argument(
        "--replay-dir",
        type=str,
        default=None,
        help="Directory containing replay JSON files",
    )
    parser.add_argument("--loop", action="store_true", help="Loop replay")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch replay directory for new episodes",
    )
    parser.add_argument("--poll", type=float, default=1.0, help="Replay dir poll interval")
    parser.add_argument(
        "--manual",
        dest="auto",
        action="store_false",
        help="Disable auto-run and use keyboard controls",
    )
    parser.add_argument(
        "--no-auto-reset",
        dest="auto_reset",
        action="store_false",
        help="Stop after an episode ends",
    )
    parser.set_defaults(auto=True, auto_reset=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.replay_dir or args.replay:
        renderer = MazeRaceReplayRenderer(cell_size=args.cell, fps=args.fps)
        try:
            if args.replay:
                data = json.loads(Path(args.replay).read_text())
                frames = data.get("frames", [])
                renderer.play_frames(frames, loop=args.loop)
            else:
                replay_dir = Path(args.replay_dir)
                replay_dir.mkdir(parents=True, exist_ok=True)
                seen: set[Path] = set()
                while renderer.running:
                    renderer.wait_screen("Waiting for recordings...")
                    files = sorted(replay_dir.glob("*.json"))
                    new_files = [path for path in files if path not in seen]
                    if not new_files:
                        if not renderer._handle_events():
                            break
                        if not args.watch:
                            break
                        time.sleep(max(args.poll, 0.1))
                        continue
                    for path in new_files:
                        if not renderer._handle_events():
                            break
                        try:
                            data = json.loads(path.read_text())
                        except json.JSONDecodeError:
                            continue
                        frames = data.get("frames", [])
                        renderer.play_frames(frames, loop=False)
                        seen.add(path)
        finally:
            renderer.close()
    else:
        cfg = MazeRaceConfig(width=args.size, height=args.size, num_envs=1, max_steps=300)
        env = MazeRaceTorchEnv(config=cfg, num_envs=1)
        MazeRaceRenderer(
            cell_size=args.cell,
            fps=args.fps,
            auto_mode=args.auto,
            auto_reset=args.auto_reset,
        ).run(env)
