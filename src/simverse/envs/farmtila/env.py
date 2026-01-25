from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import gymnasium as gym

from simverse.envs.farmtila.config import FarmtilaConfig
from simverse.envs.farmtila.agent import FarmtilaAgent

from simverse.abstractor.simenv import SimEnv

class FarmtilaEnv(SimEnv):
    HARVEST_ACTION = 4
    PICKUP_ACTION = 5
    ACTION_SPACE = gym.spaces.Discrete(6)

    @property
    def action_space(self):
        return self.ACTION_SPACE

    def __init__(self, config: FarmtilaConfig):
        self.config = config

        self.seed_grid = np.zeros((config.width, config.height))
        self.owner_grid = np.full((config.width, config.height), -1)
        self.farm_grid = np.zeros((config.width, config.height), dtype=np.uint8)

        self.agents: List[FarmtilaAgent] = []
        self.rng = np.random.default_rng()

        self.steps = 0
        self.last_pickups: List[Tuple[int, int, int]] = []
        self.seeds_spawned = 0
        self.done = False
        self.winner: FarmtilaAgent | None = None
        self.max_harvested_tiles = max(1, int(self.config.width * self.config.height * 0.4))

    def reset(self):
        self.seed_grid.fill(0)
        self.owner_grid.fill(-1)
        self.farm_grid.fill(0)
        self.agents = self._spawn_agents()
        self.steps = 0
        self.last_pickups.clear()
        self.seeds_spawned = 0
        self.done = False
        self.winner = None
        self._spawn_seeds_if_due(force=True)
        for agent in self.agents:
            agent.reset()
        return self._get_observation()

    def step(self, actions: Dict[int, int] | Iterable[int] | int | None = None):
        if self.done:
            return self._package_step_result()
        action_map = self._normalize_actions(actions)
        self.last_pickups = []
        for agent in self.agents:
            reward = -0.01
            action = action_map.get(agent.agent_id)
            if action is not None:
                dx, dy = self._action_to_delta(action)
                new_x = int(np.clip(agent.position[0] + dx, 0, self.config.width - 1))
                new_y = int(np.clip(agent.position[1] + dy, 0, self.config.height - 1))
                agent.position = (new_x, new_y)
                if action == self.HARVEST_ACTION:
                    if self._plant_farm(agent):
                        reward += 5.0
                elif action == self.PICKUP_ACTION:
                    if self._collect_seed_if_present(agent):
                        reward += 1.0
            agent.reward += reward
        self.steps += 1
        self._spawn_seeds_if_due()
        self.check_episode_end()
        return self._package_step_result()

    def step_random(self):
        actions = {
            agent.agent_id: int(self.rng.integers(0, 4))
            for agent in self.agents
        }
        return self.step(actions)

    def render(self):
        pass

    def _spawn_agents(self) -> List[FarmtilaAgent]:
        agents: List[FarmtilaAgent] = []
        occupied = set()
        for agent_id in range(self.config.num_agents):
            while True:
                x = int(self.rng.integers(0, self.config.width))
                y = int(self.rng.integers(0, self.config.height))
                if (x, y) not in occupied:
                    occupied.add((x, y))
                    break
            agents.append(
                FarmtilaAgent(
                    agent_id=agent_id,
                    position=(x, y),
                    action_space=np.arange(self.ACTION_SPACE.n, dtype=np.int64),
                )
            )
        return agents

    def _get_observation(self):
        
        # [3, width, height]
        obs = np.stack([self.seed_grid, self.owner_grid, self.farm_grid], axis=0)

        return {
            "obs": obs,
            "agents": [
                {
                    "id": agent.agent_id,
                    "position": agent.position,
                    "inventory": agent.inventory,
                    "harvested_tiles": agent.harvested_tiles,
                    "reward": agent.reward,
                }
                for agent in self.agents
            ],
            "done": self.done,
            "winner": self.winner.agent_id if self.winner else None,
            "steps": self.steps,
        }

    def get_grid_seed_random(self, *, force: bool = False, limit: int | None = None) -> List[Tuple[int, int]]:
        if self.config.spawn_seed_every <= 0 and not force:
            return []
        if not force and self.steps % self.config.spawn_seed_every != 0:
            return []
        total_cells = self.config.width * self.config.height
        if total_cells == 0 or self.config.seeds_per_spawn <= 0:
            return []
        budget = self._remaining_seed_budget()
        if budget <= 0:
            return []
        capped_limit = budget if limit is None else min(limit, budget)
        if capped_limit <= 0:
            return []
        count = min(self.config.seeds_per_spawn, total_cells, capped_limit)
        if count <= 0:
            return []
        flat_indices = self.rng.choice(total_cells, size=count, replace=False)
        positions = []
        for idx in np.atleast_1d(flat_indices):
            x = int(idx) // self.config.height
            y = int(idx) % self.config.height
            positions.append((x, y))
        return positions

    def _spawn_seeds_if_due(self, *, force: bool = False):
        positions = self.get_grid_seed_random(force=force)
        if not positions:
            return
        spawned = 0
        for x, y in positions:
            if self.seed_grid[x, y] > 0 or self.farm_grid[x, y] > 0:
                continue
            self.seed_grid[x, y] = 1
            spawned += 1
        self.seeds_spawned += spawned

    def _collect_seed_if_present(self, agent: FarmtilaAgent) -> bool:
        x, y = agent.position
        if self.seed_grid[x, y] > 0:
            self.seed_grid[x, y] = 0
            agent.inventory += 1
            self.last_pickups.append((agent.agent_id, x, y))
            return True
        return False

    def _plant_farm(self, agent: FarmtilaAgent) -> bool:
        if agent.inventory <= 0:
            return False
        x, y = agent.position
        if self.farm_grid[x, y]:
            return False
        self.farm_grid[x, y] = 1
        self.owner_grid[x, y] = agent.agent_id
        agent.inventory -= 1
        agent.harvested_tiles += 1
        return True

    def _remaining_seed_budget(self) -> int:
        return max(0, self.config.total_seeds_per_episode - self.seeds_spawned)

    def check_episode_end(self) -> bool:
        if self.steps >= self.config.max_steps:
            self.done = True
            return True
        for agent in self.agents:
            if agent.harvested_tiles >= self.max_harvested_tiles:
                self.winner = agent
                agent.reward += 20.0
                self.done = True
                return True
        if self._remaining_seed_budget() <= 0:
            self.done = True
            return True
        return False

    def _normalize_actions(self, actions: Dict[int, int] | Iterable[int] | int | None) -> Dict[int, int]:
        if actions is None:
            return {}
        if isinstance(actions, dict):
            return actions
        if isinstance(actions, int):
            return {0: actions}
        return {agent_id: action for agent_id, action in enumerate(actions)}

    def _action_to_delta(self, action: int) -> tuple[int, int]:
        return {
            0: (0, -1),
            1: (0, 1),
            2: (-1, 0),
            3: (1, 0),
        }.get(action, (0, 0))

    def _package_step_result(self):
        obs = self._get_observation()
        rewards = {agent.agent_id: agent.reward for agent in self.agents}
        for agent in self.agents:
            agent.reward = 0.0
        dones = self.done
        info = {
            "winner": self.winner.agent_id if self.winner else None,
            "steps": self.steps,
        }
        return obs, rewards, dones, info
