from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch

from simverse.abstractor.simtorch_env import SimTorchEnv
from simverse.envs.snake.agent import SnakeAgent
from simverse.envs.snake.config import SnakeConfig


class SnakeTorchEnv(SimTorchEnv):
    """Torch-native batched Snake environment for PPO training."""

    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_SPACE = gym.spaces.Discrete(4)

    WINNER_NONE = -1
    WINNER_LOSE = -2
    WINNER_WIN = 0

    def __init__(
        self,
        config: SnakeConfig,
        num_envs: int | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.config = config
        if int(self.config.num_agents) != 1:
            raise ValueError("SnakeTorchEnv supports exactly one agent")

        self.num_envs = max(1, int(num_envs or self.config.num_envs))
        self.num_agents = 1
        self.width = int(self.config.width)
        self.height = int(self.config.height)
        if self.width < 5 or self.height < 5:
            raise ValueError("Snake requires width/height >= 5")

        self.max_cells = self.width * self.height
        self.interior_cells = (self.width - 2) * (self.height - 2)
        max_straight_length = max(self.width - 2, self.height - 2)
        self.init_length = max(
            2,
            min(int(self.config.init_length), self.interior_cells, max_straight_length),
        )

        self.agents: list[SnakeAgent] = []

        self.register_buffer(
            "snake_segments",
            torch.zeros((self.num_envs, self.max_cells, 2), dtype=torch.int64),
        )
        self.register_buffer(
            "snake_length",
            torch.full((self.num_envs,), self.init_length, dtype=torch.int64),
        )
        self.register_buffer("direction", torch.zeros(self.num_envs, dtype=torch.int64))
        self.register_buffer("food_pos", torch.zeros((self.num_envs, 2), dtype=torch.int64))

        self.register_buffer("steps", torch.zeros(self.num_envs, dtype=torch.int64))
        self.register_buffer("score", torch.zeros(self.num_envs, dtype=torch.int64))
        self.register_buffer("done", torch.zeros(self.num_envs, dtype=torch.bool))
        self.register_buffer(
            "winner",
            torch.full((self.num_envs,), self.WINNER_NONE, dtype=torch.int64),
        )

        self.obs_channels = 8
        self.register_buffer(
            "obs_buffer",
            torch.zeros(
                self.num_envs,
                self.obs_channels,
                self.height,
                self.width,
                dtype=self.dtype,
            ),
        )

        wall_map = torch.zeros((1, self.height, self.width), dtype=self.dtype)
        wall_map[:, 0, :] = 1.0
        wall_map[:, self.height - 1, :] = 1.0
        wall_map[:, :, 0] = 1.0
        wall_map[:, :, self.width - 1] = 1.0
        self.register_buffer("wall_map", wall_map)

        self.register_buffer(
            "direction_deltas",
            torch.tensor(
                [
                    [0, -1],
                    [0, 1],
                    [-1, 0],
                    [1, 0],
                ],
                dtype=torch.int64,
            ),
        )
        self.register_buffer("opposite_direction", torch.tensor([1, 0, 3, 2], dtype=torch.int64))
        self.register_buffer("env_idx", torch.arange(self.num_envs, dtype=torch.int64))
        self.register_buffer("cell_idx", torch.arange(self.max_cells, dtype=torch.int64))

        self.to(self.device)

    @property
    def action_space(self):
        return self.ACTION_SPACE

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.obs_channels, self.height, self.width),
            dtype=np.float32,
        )

    def assign_agents(self, agents: list[SnakeAgent]) -> None:
        if len(agents) != 1:
            raise ValueError("SnakeTorchEnv requires exactly one agent")
        self.agents = agents

    def reset(self) -> Dict[str, torch.Tensor]:
        self._reset_indices(torch.arange(self.num_envs, device=self.device, dtype=torch.int64))
        return self._get_observation()

    def step(
        self,
        actions: torch.Tensor | Sequence[int] | np.ndarray | Dict[int, int] | None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if self.num_envs > 1:
            done_indices = torch.nonzero(self.done, as_tuple=True)[0]
            if done_indices.numel() > 0:
                # Recycle finished vectorized slots without CPU sync/tolist loops.
                self._reset_indices(done_indices)

        action_tensor = self._normalize_actions(actions)

        rewards = torch.zeros(
            (self.num_envs, self.num_agents),
            dtype=self.dtype,
            device=self.device,
        )
        active = ~self.done
        if not bool(active.any().item()):
            obs = self._get_observation()
            info = {
                "winner": self.winner.clone(),
                "steps": self.steps.clone(),
                "score": self.score.clone(),
            }
            return obs, rewards, self.done.clone(), info

        chosen_actions = action_tensor[:, 0]
        valid_actions = (chosen_actions >= 0) & (chosen_actions <= self.ACTION_RIGHT)

        next_direction = self.direction.clone()
        assign_mask = active & valid_actions
        if bool(assign_mask.any().item()):
            next_direction[assign_mask] = chosen_actions[assign_mask]

        reverse_mask = (
            active
            & valid_actions
            & (self.snake_length > 1)
            & (chosen_actions == self.opposite_direction[self.direction])
        )
        if bool(reverse_mask.any().item()):
            next_direction[reverse_mask] = self.direction[reverse_mask]

        self.direction.copy_(next_direction)

        head = self.snake_segments[:, 0, :]
        delta = self.direction_deltas[self.direction]
        new_head = head + delta

        ate_food = (
            active
            & (new_head[:, 0] == self.food_pos[:, 0])
            & (new_head[:, 1] == self.food_pos[:, 1])
        )

        collision_length = self.snake_length - (~ate_food).to(torch.int64)
        collision_length = torch.clamp(collision_length, min=0, max=self.max_cells)
        occupied_mask = self.cell_idx.unsqueeze(0) < collision_length.unsqueeze(1)
        collision_mask = (
            (self.snake_segments[:, :, 0] == new_head[:, 0].unsqueeze(1))
            & (self.snake_segments[:, :, 1] == new_head[:, 1].unsqueeze(1))
            & occupied_mask
        )
        self_collision = active & collision_mask.any(dim=1)

        wall_collision = active & (
            (new_head[:, 0] <= 0)
            | (new_head[:, 0] >= (self.width - 1))
            | (new_head[:, 1] <= 0)
            | (new_head[:, 1] >= (self.height - 1))
        )

        crashed = active & (self_collision | wall_collision)
        moved = active & (~crashed)

        moved_indices = torch.nonzero(moved, as_tuple=True)[0]
        if moved_indices.numel() > 0:
            self.snake_segments[moved_indices, 1:, :] = self.snake_segments[moved_indices, :-1, :]
            self.snake_segments[moved_indices, 0, :] = new_head[moved_indices]

            self.steps[moved_indices] += 1

            grew_mask = ate_food[moved_indices]
            if bool(grew_mask.any().item()):
                grew_indices = moved_indices[grew_mask]
                self.snake_length[grew_indices] = torch.clamp(
                    self.snake_length[grew_indices] + 1,
                    max=self.max_cells,
                )
                self.score[grew_indices] += 1
                rewards[grew_indices, 0] += float(self.config.food_reward)
                self._spawn_food_for_envs(grew_indices)

        crashed_indices = torch.nonzero(crashed, as_tuple=True)[0]
        if crashed_indices.numel() > 0:
            self.steps[crashed_indices] += 1
            rewards[crashed_indices, 0] -= float(self.config.crash_penalty)
            self.winner[crashed_indices] = self.WINNER_LOSE

        timed_out = active & (self.steps >= int(self.config.max_steps))
        if bool(timed_out.any().item()):
            self.winner[timed_out & (~crashed)] = self.WINNER_WIN

        self.done |= crashed | timed_out

        obs = self._get_observation()
        info = {
            "winner": self.winner.clone(),
            "steps": self.steps.clone(),
            "score": self.score.clone(),
        }
        return obs, rewards, self.done.clone(), info

    def get_observation(self) -> Dict[str, torch.Tensor]:
        return self._get_observation()

    def _normalize_actions(
        self,
        actions: torch.Tensor | Sequence[int] | np.ndarray | Dict[int, int] | None,
    ) -> torch.Tensor:
        if actions is None:
            return torch.full((self.num_envs, 1), -1, dtype=torch.int64, device=self.device)

        if isinstance(actions, dict):
            if self.num_envs != 1:
                raise ValueError("Dict actions are supported only when num_envs == 1")
            action_value = int(actions.get(0, -1))
            return torch.as_tensor([[action_value]], dtype=torch.int64, device=self.device)

        action_tensor = actions if isinstance(actions, torch.Tensor) else torch.as_tensor(actions)

        if action_tensor.ndim == 0:
            action_tensor = action_tensor.unsqueeze(0)
        if action_tensor.ndim == 2 and action_tensor.shape[1] == 1:
            action_tensor = action_tensor[:, 0]
        if action_tensor.ndim != 1:
            raise ValueError(
                "Expected actions with shape [num_envs] or [num_envs, 1], "
                f"got {tuple(action_tensor.shape)}"
            )

        if action_tensor.shape[0] == 1 and self.num_envs > 1:
            action_tensor = action_tensor.repeat(self.num_envs)

        if action_tensor.shape[0] != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {int(action_tensor.shape[0])}")

        action_tensor = action_tensor.to(device=self.device, dtype=torch.int64)
        invalid = (action_tensor < 0) | (action_tensor > self.ACTION_RIGHT)
        action_tensor = torch.where(invalid, torch.full_like(action_tensor, -1), action_tensor)
        return action_tensor.unsqueeze(1)

    def _reset_indices(self, indices: Sequence[int] | torch.Tensor) -> None:
        if isinstance(indices, torch.Tensor):
            env_indices = indices.to(device=self.device, dtype=torch.int64)
        else:
            index_list = list(indices)
            if not index_list:
                return
            env_indices = torch.as_tensor(index_list, device=self.device, dtype=torch.int64)

        if env_indices.numel() == 0:
            return

        env_indices = torch.unique(env_indices)
        count = int(env_indices.numel())

        self.done[env_indices] = False
        self.winner[env_indices] = self.WINNER_NONE
        self.steps[env_indices] = 0
        self.score[env_indices] = 0
        self.snake_length[env_indices] = self.init_length

        if (self.width - 2) >= self.init_length and (self.height - 2) >= self.init_length:
            directions = torch.randint(0, 4, (count,), device=self.device, dtype=torch.int64)
        elif (self.width - 2) >= self.init_length:
            # Horizontal only: left/right.
            directions = torch.randint(0, 2, (count,), device=self.device, dtype=torch.int64)
            directions = torch.where(
                directions == 0,
                torch.full_like(directions, self.ACTION_LEFT),
                torch.full_like(directions, self.ACTION_RIGHT),
            )
        else:
            # Vertical only: up/down.
            directions = torch.randint(0, 2, (count,), device=self.device, dtype=torch.int64)
            directions = torch.where(
                directions == 0,
                torch.full_like(directions, self.ACTION_UP),
                torch.full_like(directions, self.ACTION_DOWN),
            )

        self.direction[env_indices] = directions

        head_x = torch.empty((count,), device=self.device, dtype=torch.int64)
        head_y = torch.empty((count,), device=self.device, dtype=torch.int64)
        length_minus_one = self.init_length - 1

        up_mask = directions == self.ACTION_UP
        if bool(up_mask.any().item()):
            up_count = int(up_mask.sum().item())
            head_x[up_mask] = torch.randint(
                1, self.width - 1, (up_count,), device=self.device, dtype=torch.int64
            )
            head_y[up_mask] = torch.randint(
                1,
                self.height - length_minus_one - 1,
                (up_count,),
                device=self.device,
                dtype=torch.int64,
            )

        down_mask = directions == self.ACTION_DOWN
        if bool(down_mask.any().item()):
            down_count = int(down_mask.sum().item())
            head_x[down_mask] = torch.randint(
                1, self.width - 1, (down_count,), device=self.device, dtype=torch.int64
            )
            head_y[down_mask] = torch.randint(
                1 + length_minus_one,
                self.height - 1,
                (down_count,),
                device=self.device,
                dtype=torch.int64,
            )

        left_mask = directions == self.ACTION_LEFT
        if bool(left_mask.any().item()):
            left_count = int(left_mask.sum().item())
            head_x[left_mask] = torch.randint(
                1,
                self.width - length_minus_one - 1,
                (left_count,),
                device=self.device,
                dtype=torch.int64,
            )
            head_y[left_mask] = torch.randint(
                1, self.height - 1, (left_count,), device=self.device, dtype=torch.int64
            )

        right_mask = directions == self.ACTION_RIGHT
        if bool(right_mask.any().item()):
            right_count = int(right_mask.sum().item())
            head_x[right_mask] = torch.randint(
                1 + length_minus_one,
                self.width - 1,
                (right_count,),
                device=self.device,
                dtype=torch.int64,
            )
            head_y[right_mask] = torch.randint(
                1, self.height - 1, (right_count,), device=self.device, dtype=torch.int64
            )

        self.snake_segments[env_indices, :, :] = 0
        delta = self.direction_deltas[directions]
        dx = delta[:, 0]
        dy = delta[:, 1]
        for segment_idx in range(self.init_length):
            self.snake_segments[env_indices, segment_idx, 0] = head_x - segment_idx * dx
            self.snake_segments[env_indices, segment_idx, 1] = head_y - segment_idx * dy

        self._spawn_food_for_envs(env_indices)

    def _spawn_food_for_envs(self, env_indices: torch.Tensor) -> None:
        if env_indices.numel() == 0:
            return

        env_indices = env_indices.to(device=self.device, dtype=torch.int64)
        lengths = self.snake_length[env_indices]
        full_mask = lengths >= self.interior_cells

        if bool(full_mask.any().item()):
            full_envs = env_indices[full_mask]
            self.food_pos[full_envs, 0] = 1
            self.food_pos[full_envs, 1] = 1
            self.done[full_envs] = True
            self.winner[full_envs] = self.WINNER_WIN

        pending = env_indices[~full_mask]
        if pending.numel() == 0:
            return

        max_attempts = 64
        for _ in range(max_attempts):
            if pending.numel() == 0:
                break

            count = int(pending.numel())
            fx = torch.randint(1, self.width - 1, (count,), device=self.device, dtype=torch.int64)
            fy = torch.randint(1, self.height - 1, (count,), device=self.device, dtype=torch.int64)

            seg_x = self.snake_segments[pending, :, 0]
            seg_y = self.snake_segments[pending, :, 1]
            occupied_mask = self.cell_idx.unsqueeze(0) < self.snake_length[pending].unsqueeze(1)
            occupied = (
                (seg_x == fx.unsqueeze(1)) & (seg_y == fy.unsqueeze(1)) & occupied_mask
            ).any(dim=1)

            valid = ~occupied
            if bool(valid.any().item()):
                valid_envs = pending[valid]
                self.food_pos[valid_envs, 0] = fx[valid]
                self.food_pos[valid_envs, 1] = fy[valid]

            pending = pending[occupied]

        if pending.numel() == 0:
            return

        for env_index in pending.detach().cpu().tolist():
            self._spawn_food_fallback(int(env_index))

    def _spawn_food_fallback(self, env_index: int) -> None:
        length = int(self.snake_length[env_index].item())
        occupied = {
            (
                int(self.snake_segments[env_index, seg_idx, 0].item()),
                int(self.snake_segments[env_index, seg_idx, 1].item()),
            )
            for seg_idx in range(length)
        }

        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if (x, y) not in occupied:
                    self.food_pos[env_index, 0] = x
                    self.food_pos[env_index, 1] = y
                    return

        self.food_pos[env_index, 0] = 1
        self.food_pos[env_index, 1] = 1
        self.done[env_index] = True
        self.winner[env_index] = self.WINNER_WIN

    def _get_observation(self) -> Dict[str, torch.Tensor]:
        self.obs_buffer.zero_()
        self.obs_buffer[:, 0].copy_(self.wall_map.expand(self.num_envs, -1, -1))

        self.obs_buffer[self.env_idx, 1, self.food_pos[:, 1], self.food_pos[:, 0]] = 1.0

        head_x = self.snake_segments[:, 0, 0]
        head_y = self.snake_segments[:, 0, 1]
        self.obs_buffer[self.env_idx, 2, head_y, head_x] = 1.0

        body_mask = (self.cell_idx.unsqueeze(0) < self.snake_length.unsqueeze(1)) & (
            self.cell_idx.unsqueeze(0) > 0
        )
        if bool(body_mask.any().item()):
            env_ids = self.env_idx.unsqueeze(1).expand(-1, self.max_cells)
            body_x = self.snake_segments[:, :, 0]
            body_y = self.snake_segments[:, :, 1]
            self.obs_buffer[env_ids[body_mask], 3, body_y[body_mask], body_x[body_mask]] = 1.0

        for direction in range(4):
            direction_mask = self.direction == direction
            if bool(direction_mask.any().item()):
                self.obs_buffer[direction_mask, 4 + direction, :, :] = 1.0

        return {
            "obs": self.obs_buffer,
            "done": self.done.clone(),
            "winner": self.winner.clone(),
            "steps": self.steps.clone(),
            "score": self.score.clone(),
        }
