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
        self.interior_width = self.width - 2
        self.interior_height = self.height - 2
        self.interior_cells = self.interior_width * self.interior_height
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
        self.register_buffer(
            "occupied_grid",
            torch.zeros((self.num_envs, self.height, self.width), dtype=torch.bool),
        )

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
        self.register_buffer(
            "init_segment_offsets",
            torch.arange(self.init_length, dtype=torch.int64),
        )

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
        active_indices = torch.nonzero(active, as_tuple=True)[0]
        if active_indices.numel() == 0:
            obs = self._get_observation()
            info = {
                "winner": self.winner.clone(),
                "steps": self.steps.clone(),
                "score": self.score.clone(),
            }
            return obs, rewards, self.done.clone(), info

        chosen_actions = action_tensor[:, 0]
        valid_actions = (chosen_actions >= 0) & (chosen_actions <= self.ACTION_RIGHT)
        assign_mask = active & valid_actions
        next_direction = torch.where(assign_mask, chosen_actions, self.direction)

        reverse_mask = (
            assign_mask
            & (self.snake_length > 1)
            & (chosen_actions == self.opposite_direction[self.direction])
        )
        next_direction = torch.where(reverse_mask, self.direction, next_direction)
        self.direction.copy_(next_direction)

        head = self.snake_segments[:, 0, :]
        tail_idx = torch.clamp(self.snake_length - 1, min=0)
        tail_x = self.snake_segments[self.env_idx, tail_idx, 0]
        tail_y = self.snake_segments[self.env_idx, tail_idx, 1]
        delta = self.direction_deltas[self.direction]
        new_head = head + delta
        new_x = new_head[:, 0]
        new_y = new_head[:, 1]

        ate_food = active & (new_x == self.food_pos[:, 0]) & (new_y == self.food_pos[:, 1])

        wall_collision = active & (
            (new_x <= 0)
            | (new_x >= (self.width - 1))
            | (new_y <= 0)
            | (new_y >= (self.height - 1))
        )

        clamped_x = torch.clamp(new_x, 0, self.width - 1)
        clamped_y = torch.clamp(new_y, 0, self.height - 1)
        occupied_at_new = self.occupied_grid[self.env_idx, clamped_y, clamped_x]
        moving_into_tail = (new_x == tail_x) & (new_y == tail_y) & (~ate_food)
        self_collision = active & occupied_at_new & (~moving_into_tail)

        crashed = active & (self_collision | wall_collision)
        moved = active & (~crashed)
        self.steps[active_indices] += 1

        moved_indices = torch.nonzero(moved, as_tuple=True)[0]
        if moved_indices.numel() > 0:
            moved_lengths = self.snake_length[moved_indices]
            shift_cells = int(moved_lengths.max().item())
            self.snake_segments[
                moved_indices,
                1 : shift_cells + 1,
                :,
            ] = self.snake_segments[moved_indices, :shift_cells, :]
            self.snake_segments[moved_indices, 0, :] = new_head[moved_indices]

            moved_non_grow_indices = torch.nonzero(moved & (~ate_food), as_tuple=True)[0]
            if moved_non_grow_indices.numel() > 0:
                self.occupied_grid[
                    moved_non_grow_indices,
                    tail_y[moved_non_grow_indices],
                    tail_x[moved_non_grow_indices],
                ] = False

            self.occupied_grid[moved_indices, new_y[moved_indices], new_x[moved_indices]] = True

            grew_indices = torch.nonzero(moved & ate_food, as_tuple=True)[0]
            if grew_indices.numel() > 0:
                self.snake_length[grew_indices] = torch.clamp(
                    self.snake_length[grew_indices] + 1,
                    max=self.max_cells,
                )
                self.score[grew_indices] += 1
                rewards[grew_indices, 0] += float(self.config.food_reward)
                self._spawn_food_for_envs(grew_indices)

        crashed_indices = torch.nonzero(crashed, as_tuple=True)[0]
        if crashed_indices.numel() > 0:
            rewards[crashed_indices, 0] -= float(self.config.crash_penalty)
            self.winner[crashed_indices] = self.WINNER_LOSE

        timed_out = active & (self.steps >= int(self.config.max_steps))
        timed_out_winners = torch.nonzero(timed_out & (~crashed), as_tuple=True)[0]
        if timed_out_winners.numel() > 0:
            self.winner[timed_out_winners] = self.WINNER_WIN

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
        self.occupied_grid[env_indices, :, :] = False

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
        up_indices = torch.nonzero(up_mask, as_tuple=True)[0]
        if up_indices.numel() > 0:
            up_count = int(up_indices.numel())
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
        down_indices = torch.nonzero(down_mask, as_tuple=True)[0]
        if down_indices.numel() > 0:
            down_count = int(down_indices.numel())
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
        left_indices = torch.nonzero(left_mask, as_tuple=True)[0]
        if left_indices.numel() > 0:
            left_count = int(left_indices.numel())
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
        right_indices = torch.nonzero(right_mask, as_tuple=True)[0]
        if right_indices.numel() > 0:
            right_count = int(right_indices.numel())
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
        offsets = self.init_segment_offsets.unsqueeze(0)
        dx = delta[:, 0].unsqueeze(1)
        dy = delta[:, 1].unsqueeze(1)
        self.snake_segments[env_indices, : self.init_length, 0] = head_x.unsqueeze(1) - offsets * dx
        self.snake_segments[env_indices, : self.init_length, 1] = head_y.unsqueeze(1) - offsets * dy

        env_grid = env_indices.unsqueeze(1).expand(-1, self.init_length)
        seg_x = self.snake_segments[env_indices, : self.init_length, 0]
        seg_y = self.snake_segments[env_indices, : self.init_length, 1]
        self.occupied_grid[env_grid, seg_y, seg_x] = True

        self._spawn_food_for_envs(env_indices)

    def _spawn_food_for_envs(self, env_indices: torch.Tensor) -> None:
        if env_indices.numel() == 0:
            return

        env_indices = env_indices.to(device=self.device, dtype=torch.int64)
        lengths = self.snake_length[env_indices]
        full_mask = lengths >= self.interior_cells

        full_envs = env_indices[full_mask]
        if full_envs.numel() > 0:
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

            occupied = self.occupied_grid[pending, fy, fx]

            valid = ~occupied
            valid_envs = pending[valid]
            if valid_envs.numel() > 0:
                self.food_pos[valid_envs, 0] = fx[valid]
                self.food_pos[valid_envs, 1] = fy[valid]

            pending = pending[occupied]

        if pending.numel() == 0:
            return

        for env_index in pending.detach().cpu().tolist():
            self._spawn_food_fallback(int(env_index))

    def _spawn_food_fallback(self, env_index: int) -> None:
        free_cells = torch.nonzero(
            ~self.occupied_grid[env_index, 1 : self.height - 1, 1 : self.width - 1].reshape(-1),
            as_tuple=True,
        )[0]
        if free_cells.numel() > 0:
            random_idx = torch.randint(
                0,
                int(free_cells.numel()),
                (1,),
                device=self.device,
                dtype=torch.int64,
            )
            chosen = int(free_cells[random_idx].item())
            self.food_pos[env_index, 0] = (chosen % self.interior_width) + 1
            self.food_pos[env_index, 1] = (chosen // self.interior_width) + 1
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

        self.obs_buffer[:, 3].copy_(self.occupied_grid)
        self.obs_buffer[self.env_idx, 3, head_y, head_x] = 0.0
        self.obs_buffer[self.env_idx, 4 + self.direction, :, :] = 1.0

        return {
            # Return a snapshot so caller-side tensors do not alias this mutable buffer
            # across step() calls.
            "obs": self.obs_buffer.clone(),
            "done": self.done.clone(),
            "winner": self.winner.clone(),
            "steps": self.steps.clone(),
            "score": self.score.clone(),
        }
