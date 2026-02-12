from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch

from simverse.abstractor.simtorch_env import SimTorchEnv
from simverse.envs.maze_race.agent import MazeRaceAgent
from simverse.envs.maze_race.config import MazeRaceConfig


class MazeRaceTorchEnv(SimTorchEnv):
    """Simple 2-agent maze racing env. First agent to its goal wins."""

    ACTION_STAY = 0
    ACTION_UP = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3
    ACTION_RIGHT = 4
    ACTION_SPACE = gym.spaces.Discrete(5)

    WINNER_NONE = -1
    WINNER_DRAW = -2

    def __init__(
        self,
        config: MazeRaceConfig,
        num_envs: int | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.config = config
        if self.config.num_agents != 2:
            raise ValueError("MazeRaceTorchEnv requires exactly 2 agents")

        self.num_envs = num_envs or getattr(config, "num_envs", 1)
        self.num_agents = self.config.num_agents
        self.width = int(self.config.width)
        self.height = int(self.config.height)
        if self.width < 7 or self.height < 7:
            raise ValueError("MazeRace requires width/height >= 7")

        self.agents: list[MazeRaceAgent] = []

        self.start0 = (1, 1)
        self.start1 = (self.width - 2, 1)
        self.goal0 = (self.width - 2, self.height - 2)
        self.goal1 = (1, self.height - 2)

        self.register_buffer("walls", self._build_maze())
        self.register_buffer("wall_map", self.walls.to(self.dtype).unsqueeze(0))

        goal0 = torch.zeros((self.height, self.width), dtype=self.dtype)
        goal1 = torch.zeros((self.height, self.width), dtype=self.dtype)

        goal0[self.goal0[1], self.goal0[0]] = 1.0
        goal1[self.goal1[1], self.goal1[0]] = 1.0

        self.register_buffer("goal0_map", goal0.unsqueeze(0))
        self.register_buffer("goal1_map", goal1.unsqueeze(0))

        self.register_buffer("agent_pos", torch.zeros(self.num_envs, 2, 2, dtype=torch.int64))
        self.register_buffer("steps", torch.zeros(self.num_envs, dtype=torch.int64))
        self.register_buffer("done", torch.zeros(self.num_envs, dtype=torch.bool))
        self.register_buffer("winner", torch.full((self.num_envs,), self.WINNER_NONE, dtype=torch.int64))
        self.register_buffer("obs_buffer", torch.zeros(self.num_envs, 5, self.height, self.width, dtype=self.dtype))
        self.register_buffer("env_idx", torch.arange(self.num_envs, dtype=torch.int64))

        self.register_buffer("delta_x", torch.tensor([0, 0, 0, -1, 1], dtype=torch.int64))
        self.register_buffer("delta_y", torch.tensor([0, -1, 1, 0, 0], dtype=torch.int64))

        self.to(self.device)

    @property
    def action_space(self):
        return self.ACTION_SPACE

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5, self.height, self.width),
            dtype=np.float32,
        )

    def assign_agents(self, agents: list[MazeRaceAgent]) -> None:
        if len(agents) != 2:
            raise ValueError("MazeRace requires exactly 2 agents")
        self.agents = agents

    def reset(self) -> Dict[str, torch.Tensor]:
        self.agent_pos[:, 0, 0] = self.start0[0]
        self.agent_pos[:, 0, 1] = self.start0[1]
        self.agent_pos[:, 1, 0] = self.start1[0]
        self.agent_pos[:, 1, 1] = self.start1[1]
        self.steps.zero_()
        self.done.zero_()
        self.winner.fill_(self.WINNER_NONE)
        return self._get_observation()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        action_tensor = self._normalize_actions(actions)
        rewards = torch.zeros((self.num_envs, self.num_agents), dtype=self.dtype, device=self.device)
        active = ~self.done

        for agent_id in range(self.num_agents):
            action = action_tensor[:, agent_id]
            action_idx = torch.clamp(action, min=0, max=4)
            has_action = action >= 0
            move_mask = active & has_action

            px = self.agent_pos[:, agent_id, 0]
            py = self.agent_pos[:, agent_id, 1]
            nx = torch.clamp(px + self.delta_x[action_idx], 0, self.width - 1)
            ny = torch.clamp(py + self.delta_y[action_idx], 0, self.height - 1)

            blocked = self.walls[ny, nx]
            can_move = move_mask & (~blocked)

            self.agent_pos[:, agent_id, 0] = torch.where(can_move, nx, px)
            self.agent_pos[:, agent_id, 1] = torch.where(can_move, ny, py)

        rewards[active, :] -= float(self.config.step_penalty)
        self.steps[active] += 1

        p0x = self.agent_pos[:, 0, 0]
        p0y = self.agent_pos[:, 0, 1]
        p1x = self.agent_pos[:, 1, 0]
        p1y = self.agent_pos[:, 1, 1]

        reached0 = active & (p0x == self.goal0[0]) & (p0y == self.goal0[1])
        reached1 = active & (p1x == self.goal1[0]) & (p1y == self.goal1[1])

        both = reached0 & reached1
        only0 = reached0 & (~reached1)
        only1 = reached1 & (~reached0)

        if torch.any(only0):
            rewards[only0, 0] += float(self.config.win_reward)
            rewards[only0, 1] -= float(self.config.lose_penalty)
            self.winner[only0] = 0
        if torch.any(only1):
            rewards[only1, 1] += float(self.config.win_reward)
            rewards[only1, 0] -= float(self.config.lose_penalty)
            self.winner[only1] = 1
        if torch.any(both):
            rewards[both, :] += float(self.config.draw_reward)
            self.winner[both] = self.WINNER_DRAW

        finished = only0 | only1 | both

        timed_out = active & (self.steps >= int(self.config.max_steps))
        draw_timeout = timed_out & (~finished)
        if torch.any(draw_timeout):
            rewards[draw_timeout, :] += float(self.config.draw_reward)
            self.winner[draw_timeout] = self.WINNER_DRAW

        self.done |= finished | timed_out

        obs = self._get_observation()
        info = {
            "winner": self.winner.clone(),
            "steps": self.steps.clone(),
        }
        return obs, rewards, self.done.clone(), info

    def _normalize_actions(self, actions: torch.Tensor | None) -> torch.Tensor:
        if actions is None:
            return torch.full(
                (self.num_envs, self.num_agents),
                -1,
                dtype=torch.int64,
                device=self.device,
            )
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        if actions.ndim == 1:
            if self.num_envs == 1 and actions.numel() == self.num_agents:
                actions = actions.unsqueeze(0)
            elif self.num_agents == 1 and actions.numel() == self.num_envs:
                actions = actions.unsqueeze(1)
        expected_shape = (self.num_envs, self.num_agents)
        if tuple(actions.shape) != expected_shape:
            raise ValueError(f"Expected actions shape {expected_shape}, got {tuple(actions.shape)}")
        return actions.to(device=self.device, dtype=torch.int64)

    def _build_maze(self) -> torch.Tensor:
        walls = torch.zeros((self.height, self.width), dtype=torch.bool)

        walls[0, :] = True
        walls[self.height - 1, :] = True
        walls[:, 0] = True
        walls[:, self.width - 1] = True

        for x in range(3, self.width - 2, 3):
            walls[1 : self.height - 1, x] = True
            gap0 = 1 + ((2 * x + 1) % (self.height - 2))
            gap1 = 1 + ((3 * x + 2) % (self.height - 2))
            walls[gap0, x] = False
            walls[gap1, x] = False

        walls[self.start0[1], self.start0[0]] = False
        walls[self.start1[1], self.start1[0]] = False
        walls[self.goal0[1], self.goal0[0]] = False
        walls[self.goal1[1], self.goal1[0]] = False
        return walls

    def _get_observation(self) -> Dict[str, torch.Tensor]:
        self.obs_buffer.zero_()

        self.obs_buffer[:, 0].copy_(self.wall_map.expand(self.num_envs, -1, -1))
        self.obs_buffer[:, 1].copy_(self.goal0_map.expand(self.num_envs, -1, -1))
        self.obs_buffer[:, 2].copy_(self.goal1_map.expand(self.num_envs, -1, -1))

        self.obs_buffer[self.env_idx, 3, self.agent_pos[:, 0, 1], self.agent_pos[:, 0, 0]] = 1.0
        self.obs_buffer[self.env_idx, 4, self.agent_pos[:, 1, 1], self.agent_pos[:, 1, 0]] = 1.0

        return {
            "obs": self.obs_buffer,
            "done": self.done.clone(),
            "winner": self.winner.clone(),
            "steps": self.steps.clone(),
        }
