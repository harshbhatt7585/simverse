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
        self.max_harvested_tiles = max(1, int(self.width * self.height * 0.4))

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
        self.register_buffer(
            "seeds_spawned",
            torch.zeros(self.num_envs, dtype=torch.int64),
        )
        self.register_buffer("steps", torch.zeros(self.num_envs, dtype=torch.int64))
        self.register_buffer("done", torch.zeros(self.num_envs, dtype=torch.bool))
        self.register_buffer(
            "winner",
            torch.full((self.num_envs,), -1, dtype=torch.int64),
        )
        self.to(self.device)

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
        self.seeds_spawned.zero_()
        self.steps.zero_()
        self.done.zero_()
        self.winner.fill_(-1)
        self._spawn_agents()
        self._spawn_seeds_if_due(force=True)
        return self._get_observation()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        action_tensor = self._normalize_actions(actions)
        rewards = torch.zeros(
            (self.num_envs, self.num_agents),
            dtype=self.dtype,
            device=self.device,
        )
        active_mask = ~self.done
        if not active_mask.any():
            obs = self._get_observation()
            return obs, rewards, self.done.clone(), {"winner": self.winner.clone()}
        rewards[active_mask] = -0.01

        delta_x = torch.tensor([0, 0, -1, 1, 0, 0], device=self.device)
        delta_y = torch.tensor([-1, 1, 0, 0, 0, 0], device=self.device)
        env_idx = torch.arange(self.num_envs, device=self.device)

        for agent_id in range(self.num_agents):
            action = action_tensor[:, agent_id]
            has_action = action >= 0
            active_action = has_action & active_mask
            action_index = torch.clamp(action, min=0, max=5)
            dx = delta_x[action_index] * active_action.to(delta_x.dtype)
            dy = delta_y[action_index] * active_action.to(delta_y.dtype)
            pos_x = self.agent_pos[:, agent_id, 0]
            pos_y = self.agent_pos[:, agent_id, 1]
            new_x = torch.clamp(pos_x + dx, 0, self.width - 1)
            new_y = torch.clamp(pos_y + dy, 0, self.height - 1)
            pos_x = torch.where(active_action, new_x, pos_x)
            pos_y = torch.where(active_action, new_y, pos_y)
            self.agent_pos[:, agent_id, 0] = pos_x
            self.agent_pos[:, agent_id, 1] = pos_y

            harvest_action = (action == self.HARVEST_ACTION) & active_action
            inventory_before = self.inventory[:, agent_id]
            harvestable = harvest_action & (inventory_before > 0)

            if harvestable.any():
                farm_level = self.farm_grid[env_idx, pos_x, pos_y]
                owner = self.owner_grid[env_idx, pos_x, pos_y]
                same_owner = owner == agent_id
                has_farm = farm_level > 0

                fortify = harvestable & has_farm & same_owner & (farm_level < self.max_farm_level)
                if fortify.any():
                    idx = env_idx[fortify]
                    self.farm_grid[idx, pos_x[fortify], pos_y[fortify]] += 1
                    self.inventory[fortify, agent_id] -= 1

                reduce_fort = harvestable & has_farm & (~same_owner) & (farm_level > 1)
                if reduce_fort.any():
                    idx = env_idx[reduce_fort]
                    self.farm_grid[idx, pos_x[reduce_fort], pos_y[reduce_fort]] -= 1
                    self.inventory[reduce_fort, agent_id] -= 1
                    rewards[reduce_fort, agent_id] -= 5.0

                steal = harvestable & has_farm & (~same_owner) & (farm_level == 1)
                if steal.any():
                    idx = env_idx[steal]
                    self.owner_grid[idx, pos_x[steal], pos_y[steal]] = agent_id
                    self.inventory[steal, agent_id] -= 1
                    self.harvested_tiles[steal, agent_id] += 1
                    rewards[steal, agent_id] += 5.0
                    prev_owner = owner[steal]
                    self.harvested_tiles[idx, prev_owner] = torch.clamp(
                        self.harvested_tiles[idx, prev_owner] - 1, min=0
                    )

                plant = harvestable & (farm_level == 0)
                if plant.any():
                    idx = env_idx[plant]
                    self.farm_grid[idx, pos_x[plant], pos_y[plant]] = 1
                    self.owner_grid[idx, pos_x[plant], pos_y[plant]] = agent_id
                    self.inventory[plant, agent_id] -= 1
                    self.harvested_tiles[plant, agent_id] += 1
                    rewards[plant, agent_id] += 5.0

            pickup = active_action & (self.seed_grid[env_idx, pos_x, pos_y] > 0)
            if pickup.any():
                idx = env_idx[pickup]
                self.seed_grid[idx, pos_x[pickup], pos_y[pickup]] = 0
                self.inventory[pickup, agent_id] += 1
                rewards[pickup, agent_id] += 1.0

        self.steps[active_mask] += 1
        self._spawn_seeds_if_due()
        self._check_episode_end(rewards)

        obs = self._get_observation()
        info = {"winner": self.winner.clone(), "steps": self.steps.clone()}
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
        if actions.shape != (self.num_envs, self.num_agents):
            raise ValueError(
                f"Expected actions shape {(self.num_envs, self.num_agents)}, got {actions.shape}"
            )
        return actions.to(device=self.device, dtype=torch.int64)

    def _spawn_agents(self) -> None:
        positions = torch.stack(
            (
                torch.randint(0, self.width, (self.num_envs, self.num_agents)),
                torch.randint(0, self.height, (self.num_envs, self.num_agents)),
            ),
            dim=-1,
        )
        self.agent_pos.copy_(positions.to(self.device))

    def _spawn_seeds_if_due(self, *, force: bool = False) -> None:
        if self.config.spawn_seed_every <= 0 and not force:
            return
        due_mask = (force | ((self.steps % self.config.spawn_seed_every) == 0)) & (~self.done)
        if not due_mask.any():
            return
        total_cells = self.width * self.height
        if total_cells == 0 or self.config.seeds_per_spawn <= 0:
            return
        env_indices = torch.arange(self.num_envs, device=self.device)[due_mask]
        for env_id in env_indices.tolist():
            budget = self._remaining_seed_budget(env_id)
            if budget <= 0:
                continue
            count = min(self.config.seeds_per_spawn, total_cells, int(budget))
            if count <= 0:
                continue
            flat_indices = torch.randperm(total_cells, device=self.device)[:count]
            xs = flat_indices // self.height
            ys = flat_indices % self.height
            empty = (self.seed_grid[env_id, xs, ys] == 0) & (self.farm_grid[env_id, xs, ys] == 0)
            if empty.any():
                self.seed_grid[env_id, xs[empty], ys[empty]] = 1
                self.seeds_spawned[env_id] += int(empty.sum().item())

    def _remaining_seed_budget(self, env_id: int) -> int:
        return max(0, int(self.config.total_seeds_per_episode - self.seeds_spawned[env_id].item()))

    def _check_episode_end(self, rewards: torch.Tensor) -> None:
        active_mask = ~self.done
        if not active_mask.any():
            return
        max_steps_mask = self.steps >= self.config.max_steps
        self.done |= max_steps_mask

        reached_harvest = self.harvested_tiles >= self.max_harvested_tiles
        harvest_winner = reached_harvest.any(dim=1)
        new_winners = harvest_winner & active_mask
        if new_winners.any():
            winner_ids = torch.argmax(self.harvested_tiles, dim=1)
            self.winner[new_winners] = winner_ids[new_winners]
            env_ids = torch.arange(self.num_envs, device=self.device)[new_winners]
            rewards[env_ids, winner_ids[new_winners]] += 20.0
            self.done |= new_winners

        budgets = self.config.total_seeds_per_episode - self.seeds_spawned
        exhausted = budgets <= 0
        self.done |= exhausted

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
