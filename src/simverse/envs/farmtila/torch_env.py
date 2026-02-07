from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch

from simverse.abstractor.simtorch_env import SimTorchEnv
from simverse.envs.farmtila.config import FarmtilaConfig


class FarmtilaTorchEnv(SimTorchEnv):
    HARVEST_ACTION = 4
    PICKUP_ACTION = 5
    ACTION_SPACE = gym.spaces.Discrete(6)

    def __init__(
        self,
        config: FarmtilaConfig,
        num_envs: int | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.config = config
        self.num_envs = num_envs or getattr(config, "num_envs", 1)
        self.width = config.width
        self.height = config.height
        self.num_agents = config.num_agents
        self.max_farm_level = max(1, getattr(config, "max_farm_level", 1))

        self.register_buffer(
            "seed_grid",
            torch.zeros(self.num_envs, self.width, self.height, dtype=torch.int64),
        )
        self.register_buffer(
            "owner_grid",
            torch.full(
                (self.num_envs, self.width, self.height),
                -1,
                dtype=torch.int64,
            ),
        )
        self.register_buffer(
            "farm_grid",
            torch.zeros(self.num_envs, self.width, self.height, dtype=torch.int64),
        )
        self.register_buffer(
            "agent_pos",
            torch.zeros(self.num_envs, self.num_agents, 2, dtype=torch.int64),
        )
        self.register_buffer(
            "inventory",
            torch.zeros(self.num_envs, self.num_agents, dtype=torch.int64),
        )
        self.register_buffer(
            "harvested_tiles",
            torch.zeros(self.num_envs, self.num_agents, dtype=torch.int64),
        )
        self.register_buffer("steps", torch.zeros(self.num_envs, dtype=torch.int64))
        self.register_buffer("done", torch.zeros(self.num_envs, dtype=torch.bool))
        self.register_buffer(
            "winner",
            torch.full((self.num_envs,), -1, dtype=torch.int64),
        )

    @property
    def action_space(self):
        return self.ACTION_SPACE

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=-1,
            high=max(self.num_agents, self.max_farm_level, 1),
            shape=(5, self.width, self.height),
            dtype=np.float32,
        )

    def reset(self) -> Dict[str, torch.Tensor]:
        self.seed_grid.zero_()
        self.owner_grid.fill_(-1)
        self.farm_grid.zero_()
        self.inventory.zero_()
        self.harvested_tiles.zero_()
        self.steps.zero_()
        self.done.zero_()
        self.winner.fill_(-1)
        self._spawn_agents()
        return self._get_observation()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError("FarmtilaTorchEnv.step is not implemented yet")

    def _spawn_agents(self) -> None:
        positions = torch.stack(
            (
                torch.randint(0, self.width, (self.num_envs, self.num_agents)),
                torch.randint(0, self.height, (self.num_envs, self.num_agents)),
            ),
            dim=-1,
        )
        self.agent_pos.copy_(positions)

    def _get_observation(self) -> Dict[str, torch.Tensor]:
        agent_grid = torch.zeros(
            (self.num_envs, self.width, self.height),
            dtype=self.dtype,
            device=self.device,
        )
        inventory_grid = torch.zeros_like(agent_grid)

        env_idx = torch.arange(self.num_envs, device=self.device).unsqueeze(1)
        for agent_id in range(self.num_agents):
            x = self.agent_pos[:, agent_id, 0]
            y = self.agent_pos[:, agent_id, 1]
            agent_grid[env_idx, x, y] = float(agent_id + 1)
            inventory_grid[env_idx, x, y] = self.inventory[:, agent_id].to(self.dtype)

        obs = torch.stack(
            [
                self.seed_grid.to(self.dtype),
                self.owner_grid.to(self.dtype),
                self.farm_grid.to(self.dtype),
                agent_grid,
                inventory_grid,
            ],
            dim=1,
        )

        return {
            "obs": obs,
            "done": self.done.clone(),
            "winner": self.winner.clone(),
            "steps": self.steps.clone(),
        }
