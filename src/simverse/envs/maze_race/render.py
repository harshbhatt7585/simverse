from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    _src = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_src))

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
        pending_actions = torch.zeros((1, 2), dtype=torch.int64, device=env.device)

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
            p0x = int(env.agent_pos[0, 0, 0].item())
            p0y = int(env.agent_pos[0, 0, 1].item())
            p1x = int(env.agent_pos[0, 1, 0].item())
            p1y = int(env.agent_pos[0, 1, 1].item())

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

            g0 = pygame.Rect(
                env.goal0[0] * self.cell_size,
                env.goal0[1] * self.cell_size,
                self.cell_size,
                self.cell_size,
            )
            g1 = pygame.Rect(
                env.goal1[0] * self.cell_size,
                env.goal1[1] * self.cell_size,
                self.cell_size,
                self.cell_size,
            )
            pygame.draw.rect(screen, self.colors["goal0"], g0)
            pygame.draw.rect(screen, self.colors["goal1"], g1)

            r = max(5, self.cell_size // 3)
            pygame.draw.circle(
                screen,
                self.colors["agent0"],
                (
                    p0x * self.cell_size + self.cell_size // 2,
                    p0y * self.cell_size + self.cell_size // 2,
                ),
                r,
            )
            pygame.draw.circle(
                screen,
                self.colors["agent1"],
                (
                    p1x * self.cell_size + self.cell_size // 2,
                    p1y * self.cell_size + self.cell_size // 2,
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
            hud = (
                f"steps={steps} | {status} | {auto_status} | "
                "arrows=blue, WASD=orange, T=toggle auto, R=reset"
            )
            text = font.render(hud, True, self.colors["text"])
            screen.blit(text, (8, env.height * self.cell_size + 12))

            pygame.display.flip()
            clock.tick(self.fps)

        pygame.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render MazeRaceTorchEnv")
    parser.add_argument("--size", type=int, default=15, help="Maze width/height (square)")
    parser.add_argument("--cell", type=int, default=36, help="Cell size in pixels")
    parser.add_argument("--fps", type=int, default=20, help="Render FPS")
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
    cfg = MazeRaceConfig(width=args.size, height=args.size, num_envs=1, max_steps=300)
    env = MazeRaceTorchEnv(config=cfg, num_envs=1)
    MazeRaceRenderer(
        cell_size=args.cell,
        fps=args.fps,
        auto_mode=args.auto,
        auto_reset=args.auto_reset,
    ).run(env)
