from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch

from simverse.abstractor.simtorch_env import SimTorchEnv
from simverse.envs.shape_draw.agent import ShapeDrawAgent
from simverse.envs.shape_draw.config import ShapeDrawConfig


class ShapeDrawTorchEnv(SimTorchEnv):
    """Single-agent shape drawing environment with batched torch stepping."""

    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_TOGGLE_PEN = 4
    ACTION_BRUSH_UP = 5
    ACTION_BRUSH_DOWN = 6
    ACTION_SPACE = gym.spaces.Discrete(7)

    def __init__(
        self,
        config: ShapeDrawConfig,
        num_envs: int | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.config = config
        if self.config.num_agents != 1:
            raise ValueError("ShapeDrawTorchEnv currently supports exactly 1 agent")

        self.num_envs = num_envs or getattr(config, "num_envs", 1)
        self.num_agents = self.config.num_agents
        self.width = self.config.width
        self.height = self.config.height
        self.agents: list[ShapeDrawAgent] = []

        self.register_buffer("canvas", torch.zeros(self.num_envs, 1, self.height, self.width))
        self.register_buffer("target", torch.zeros(self.num_envs, 1, self.height, self.width))
        self.register_buffer("pen_x", torch.zeros(self.num_envs, dtype=torch.int64))
        self.register_buffer("pen_y", torch.zeros(self.num_envs, dtype=torch.int64))
        self.register_buffer("pen_down", torch.zeros(self.num_envs, dtype=torch.bool))
        self.register_buffer(
            "brush",
            torch.full((self.num_envs,), 2, dtype=torch.int64),
        )
        self.register_buffer("steps", torch.zeros(self.num_envs, dtype=torch.int64))
        self.register_buffer("done", torch.zeros(self.num_envs, dtype=torch.bool))
        self.register_buffer("winner", torch.full((self.num_envs,), -1, dtype=torch.int64))
        self.register_buffer("similarity", torch.zeros(self.num_envs, dtype=self.dtype))

        self.register_buffer("x_coords", torch.arange(self.width, dtype=torch.int64))
        self.register_buffer("y_coords", torch.arange(self.height, dtype=torch.int64))
        self.register_buffer("env_idx", torch.arange(self.num_envs, dtype=torch.int64))

        self.to(self.device)

    @property
    def action_space(self):
        return self.ACTION_SPACE

    @property
    def observation_space(self):
        # canvas, target, pen marker
        return gym.spaces.Box(
            low=0.0, high=1.0, shape=(3, self.height, self.width), dtype=np.float32
        )

    def assign_agents(self, agents: list[ShapeDrawAgent]) -> None:
        if len(agents) != 1:
            raise ValueError("ShapeDraw supports exactly 1 agent")
        self.agents = agents

    def reset(self) -> Dict[str, torch.Tensor]:
        self.canvas.zero_()
        self.target = self._sample_targets()
        self.pen_x = torch.randint(0, self.width, (self.num_envs,), device=self.device)
        self.pen_y = torch.randint(0, self.height, (self.num_envs,), device=self.device)
        self.pen_down.zero_()
        self.brush.fill_(2)
        self.steps.zero_()
        self.done.zero_()
        self.winner.fill_(-1)
        self.similarity = self._similarity()
        return self._get_observation()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        action_tensor = self._normalize_actions(actions)
        rewards = torch.zeros(
            (self.num_envs, self.num_agents), dtype=self.dtype, device=self.device
        )
        active = ~self.done

        prev_similarity = self.similarity.clone()

        move_action = action_tensor[:, 0]
        self.pen_y = torch.where(
            active & (move_action == self.ACTION_UP), self.pen_y - 1, self.pen_y
        )
        self.pen_y = torch.where(
            active & (move_action == self.ACTION_DOWN),
            self.pen_y + 1,
            self.pen_y,
        )
        self.pen_x = torch.where(
            active & (move_action == self.ACTION_LEFT),
            self.pen_x - 1,
            self.pen_x,
        )
        self.pen_x = torch.where(
            active & (move_action == self.ACTION_RIGHT),
            self.pen_x + 1,
            self.pen_x,
        )

        self.pen_x = torch.clamp(self.pen_x, 0, self.width - 1)
        self.pen_y = torch.clamp(self.pen_y, 0, self.height - 1)

        toggles = active & (move_action == self.ACTION_TOGGLE_PEN)
        self.pen_down = torch.where(toggles, ~self.pen_down, self.pen_down)

        brush_up = active & (move_action == self.ACTION_BRUSH_UP)
        brush_down = active & (move_action == self.ACTION_BRUSH_DOWN)
        self.brush = torch.where(brush_up, self.brush + 1, self.brush)
        self.brush = torch.where(brush_down, self.brush - 1, self.brush)
        self.brush = torch.clamp(self.brush, self.config.min_brush, self.config.max_brush)

        draw_mask = active & self.pen_down
        if torch.any(draw_mask):
            self._draw_disks(draw_mask)

        self.steps[active] += 1
        self.similarity = self._similarity()

        delta = self.similarity - prev_similarity
        rewards[:, 0] = delta
        rewards[:, 0] -= float(self.config.step_penalty)
        rewards[:, 0] -= self.pen_down.to(self.dtype) * float(self.config.draw_penalty)

        solved = self.similarity >= float(self.config.completion_threshold)
        timed_out = self.steps >= self.config.max_steps
        finished_now = (~self.done) & (solved | timed_out)
        self.done |= finished_now

        if torch.any(solved & finished_now):
            rewards[solved & finished_now, 0] += float(self.config.completion_bonus)

        obs = self._get_observation()
        info = {
            "similarity": self.similarity.clone(),
            "pen_down": self.pen_down.clone(),
            "brush": self.brush.clone(),
            "steps": self.steps.clone(),
        }
        return obs, rewards, self.done.clone(), info

    def _normalize_actions(self, actions: torch.Tensor | None) -> torch.Tensor:
        if actions is None:
            return torch.full(
                (self.num_envs, self.num_agents), -1, dtype=torch.int64, device=self.device
            )
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        if actions.ndim == 1:
            if self.num_envs == 1 and actions.numel() == self.num_agents:
                actions = actions.unsqueeze(0)
            elif self.num_agents == 1 and actions.numel() == self.num_envs:
                actions = actions.unsqueeze(1)
        expected_shape = (self.num_envs, self.num_agents)
        actual_shape = tuple(actions.shape)
        if actual_shape != expected_shape:
            raise ValueError(f"Expected actions shape {expected_shape}, got {actual_shape}")
        return actions.to(device=self.device, dtype=torch.int64)

    def _sample_targets(self) -> torch.Tensor:
        targets = torch.zeros(
            (self.num_envs, 1, self.height, self.width), dtype=self.dtype, device=self.device
        )
        for env_idx in range(self.num_envs):
            shape_type = int(torch.randint(0, 3, (1,), device=self.device).item())
            cx = int(
                torch.randint(self.width // 4, self.width * 3 // 4, (1,), device=self.device).item()
            )
            cy = int(
                torch.randint(
                    self.height // 4, self.height * 3 // 4, (1,), device=self.device
                ).item()
            )
            size = int(
                torch.randint(
                    max(6, self.width // 8), max(7, self.width // 4), (1,), device=self.device
                ).item()
            )

            if shape_type == 0:
                mask = self._circle_mask(cx, cy, size)
            elif shape_type == 1:
                mask = self._square_mask(cx, cy, size)
            else:
                mask = self._triangle_mask(cx, cy, size)

            targets[env_idx, 0][mask] = 1.0
        return targets

    def _circle_mask(self, cx: int, cy: int, radius: int) -> torch.Tensor:
        xx = self.x_coords.view(1, -1).expand(self.height, -1)
        yy = self.y_coords.view(-1, 1).expand(-1, self.width)
        return (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy) <= radius * radius

    def _square_mask(self, cx: int, cy: int, half: int) -> torch.Tensor:
        x0 = max(0, cx - half)
        x1 = min(self.width, cx + half)
        y0 = max(0, cy - half)
        y1 = min(self.height, cy + half)
        mask = torch.zeros((self.height, self.width), dtype=torch.bool, device=self.device)
        mask[y0:y1, x0:x1] = True
        return mask

    def _triangle_mask(self, cx: int, cy: int, size: int) -> torch.Tensor:
        top = max(0, cy - size)
        bottom = min(self.height - 1, cy + size)
        mask = torch.zeros((self.height, self.width), dtype=torch.bool, device=self.device)
        span = max(bottom - top, 1)
        for y in range(top, bottom + 1):
            frac = (y - top) / span
            half_width = int(size * frac)
            x0 = max(0, cx - half_width)
            x1 = min(self.width, cx + half_width + 1)
            mask[y, x0:x1] = True
        return mask

    def _draw_disks(self, draw_mask: torch.Tensor) -> None:
        active_envs = self.env_idx[draw_mask]
        for env_idx in active_envs.tolist():
            px = int(self.pen_x[env_idx].item())
            py = int(self.pen_y[env_idx].item())
            r = int(self.brush[env_idx].item())
            x0 = max(0, px - r)
            x1 = min(self.width - 1, px + r)
            y0 = max(0, py - r)
            y1 = min(self.height - 1, py + r)
            if x0 > x1 or y0 > y1:
                continue
            xs = torch.arange(x0, x1 + 1, device=self.device)
            ys = torch.arange(y0, y1 + 1, device=self.device)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            disk = (grid_x - px) * (grid_x - px) + (grid_y - py) * (grid_y - py) <= r * r
            patch = self.canvas[env_idx, 0, y0 : y1 + 1, x0 : x1 + 1]
            patch[disk] = 1.0

    def _similarity(self) -> torch.Tensor:
        # 1 - mean absolute difference in [0, 1]
        diff = torch.abs(self.target - self.canvas)
        return 1.0 - diff.mean(dim=(1, 2, 3)).to(self.dtype)

    def _get_observation(self) -> Dict[str, torch.Tensor]:
        pen_map = torch.zeros_like(self.canvas)
        pen_map[self.env_idx, 0, self.pen_y, self.pen_x] = 1.0
        obs = torch.cat((self.canvas, self.target, pen_map), dim=1).to(self.dtype)
        return {
            "obs": obs,
            "done": self.done.clone(),
            "winner": self.winner.clone(),
            "steps": self.steps.clone(),
        }
