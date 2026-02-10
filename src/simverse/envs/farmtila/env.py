from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Tuple

import gymnasium as gym
import numpy as np

from simverse.abstractor.simenv import SimEnv
from simverse.abstractor.simvector_env import SimVectorEnv
from simverse.envs.farmtila.agent import FarmtilaAgent
from simverse.envs.farmtila.config import FarmtilaConfig


class FarmtilaEnv(SimEnv):
    HARVEST_ACTION = 4
    PICKUP_ACTION = 5
    ACTION_SPACE = gym.spaces.Discrete(6)
    LAND_EMPTY = 0
    LAND_TERRITORY_LOCKED = 1
    LAND_TERRITORY_UNLOCKED = 2
    LAND_HARVESTED = 3

    @property
    def action_space(self):
        return self.ACTION_SPACE

    @property
    def observation_space(self):
        # 5 channels: seed_grid, owner_grid, land_grid, agent_grid, inventory_grid
        return gym.spaces.Box(
            low=-1,
            high=max(self.config.num_agents, self.LAND_HARVESTED, 1),
            shape=(5, self.config.width, self.config.height),
            dtype=np.float32,
        )

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
        self.territory_block_size = max(2, int(getattr(self.config, "territory_block_size", 3)))

    def reset(self):
        self.seed_grid.fill(0)
        self.owner_grid.fill(-1)
        self.farm_grid.fill(0)
        if not self.agents:
            self.agents = self._spawn_agents()
        else:
            self._reset_agent_positions()
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
            reward = -float(getattr(self.config, "step_cost", 0.005))
            action = action_map.get(agent.agent_id)
            if action is not None:
                dx, dy = self._action_to_delta(action)
                new_x = int(np.clip(agent.position[0] + dx, 0, self.config.width - 1))
                new_y = int(np.clip(agent.position[1] + dy, 0, self.config.height - 1))
                agent.position = (new_x, new_y)
                reward += self._seed_proximity_reward(agent.position)
                if action == self.HARVEST_ACTION:
                    land_reward, waive_step_cost = self._land_action(agent)
                    reward += land_reward
                    if waive_step_cost:
                        reward += float(getattr(self.config, "step_cost", 0.005))
                if self._collect_seed_if_present(agent):
                    reward += 1.0
            agent.reward += reward
        self.steps += 1
        self._spawn_seeds_if_due()
        self.check_episode_end()
        return self._package_step_result()

    def step_random(self):
        actions = {agent.agent_id: int(self.rng.integers(0, 4)) for agent in self.agents}
        return self.step(actions)

    def render(self):
        pass

    def get_observation(self):
        """Public method required by SimEnv abstract class."""
        return self._get_observation()

    def assign_agents(self, agents: List[FarmtilaAgent]) -> None:
        self.agents = agents

    def _spawn_agents(self) -> List[FarmtilaAgent]:
        agents: List[FarmtilaAgent] = []
        for agent_id, (x, y) in enumerate(self._sample_unique_positions(self.config.num_agents)):
            agents.append(
                FarmtilaAgent(
                    agent_id=agent_id,
                    position=(x, y),
                    action_space=np.arange(self.ACTION_SPACE.n, dtype=np.int64),
                )
            )
        return agents

    def _reset_agent_positions(self) -> None:
        for agent, (x, y) in zip(self.agents, self._sample_unique_positions(len(self.agents))):
            agent.position = (x, y)

    def _sample_unique_positions(self, count: int) -> List[Tuple[int, int]]:
        occupied = set()
        positions: List[Tuple[int, int]] = []
        for _ in range(count):
            while True:
                x = int(self.rng.integers(0, self.config.width))
                y = int(self.rng.integers(0, self.config.height))
                if (x, y) not in occupied:
                    occupied.add((x, y))
                    positions.append((x, y))
                    break
        return positions

    def _get_observation(self):
        # Build agent position grid: 0 = no agent, agent_id + 1 = agent present
        agent_grid = np.zeros((self.config.width, self.config.height), dtype=np.float32)
        inventory_grid = np.zeros((self.config.width, self.config.height), dtype=np.float32)
        for agent in self.agents:
            x, y = agent.position
            agent_grid[x, y] = agent.agent_id + 1  # +1 so 0 means empty
            inventory_grid[x, y] = agent.inventory

        # [5, width, height]: seed_grid, owner_grid, land_grid, agent_grid, inventory_grid
        obs = np.stack(
            [
                self.seed_grid.astype(np.float32),
                self.owner_grid.astype(np.float32),
                self.farm_grid.astype(np.float32),
                agent_grid,
                inventory_grid,
            ],
            axis=0,
        )

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

    def get_grid_seed_random(
        self, *, force: bool = False, limit: int | None = None
    ) -> List[Tuple[int, int]]:
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
            if self.seed_grid[x, y] > 0 or self.farm_grid[x, y] > self.LAND_EMPTY:
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

    def _nearest_seed_distance(self, position: Tuple[int, int]) -> int | None:
        seed_positions = np.argwhere(self.seed_grid > 0)
        if seed_positions.size == 0:
            return None
        pos_x, pos_y = position
        distances = np.abs(seed_positions[:, 0] - pos_x) + np.abs(seed_positions[:, 1] - pos_y)
        return int(distances.min())

    def _seed_proximity_reward(self, current_position: Tuple[int, int]) -> float:
        step_scale = float(getattr(self.config, "seed_proximity_reward_per_step", 0.0))
        if step_scale <= 0.0:
            return 0.0
        current_distance = self._nearest_seed_distance(current_position)
        if current_distance is None:
            return 0.0
        max_distance = max(1, (self.config.width - 1) + (self.config.height - 1))
        closeness = (max_distance - current_distance) / max_distance
        return float(np.clip(closeness, 0.0, 1.0) * step_scale)

    def _land_action(self, agent: FarmtilaAgent) -> tuple[float, bool]:
        if agent.inventory <= 0:
            return 0.0, False

        reward = 0.0
        waive_step_cost = False
        x, y = agent.position
        current_state = int(self.farm_grid[x, y])
        current_owner = int(self.owner_grid[x, y])

        # Stage 1: spend seed to create territory (locked).
        if current_state == self.LAND_EMPTY:
            agent.inventory -= 1
            self.owner_grid[x, y] = agent.agent_id
            self.farm_grid[x, y] = self.LAND_TERRITORY_LOCKED
            reward += float(getattr(self.config, "territory_claim_reward", 0.1))
            if bool(
                getattr(self.config, "adjacent_territory_step_cost_waiver", True)
            ) and self._is_adjacent_harvested(agent.agent_id, x, y):
                waive_step_cost = True
            unlocked_blocks = self._unlock_completed_territory_blocks(agent.agent_id, x, y)
            if unlocked_blocks > 0:
                reward += unlocked_blocks * float(
                    getattr(self.config, "territory_unlock_reward", 5.0)
                )
            return reward, waive_step_cost

        # Stage 2: spend seed on unlocked territory to harvest one land tile.
        if current_owner == agent.agent_id and current_state == self.LAND_TERRITORY_UNLOCKED:
            agent.inventory -= 1
            self.farm_grid[x, y] = self.LAND_HARVESTED
            agent.harvested_tiles += 1
            reward += float(getattr(self.config, "harvest_on_unlocked_reward", 1.0))
            return reward, False

        return 0.0, False

    def _unlock_completed_territory_blocks(self, agent_id: int, x: int, y: int) -> int:
        size = self.territory_block_size
        unlocked = 0
        x_start = max(0, x - size + 1)
        x_end = min(x, self.config.width - size)
        y_start = max(0, y - size + 1)
        y_end = min(y, self.config.height - size)

        for sx in range(x_start, x_end + 1):
            for sy in range(y_start, y_end + 1):
                owner_block = self.owner_grid[sx : sx + size, sy : sy + size]
                land_block = self.farm_grid[sx : sx + size, sy : sy + size]
                owned = np.all(owner_block == agent_id)
                usable = np.all(land_block >= self.LAND_TERRITORY_LOCKED)
                has_locked = np.any(land_block == self.LAND_TERRITORY_LOCKED)
                if not (owned and usable and has_locked):
                    continue
                lock_mask = land_block == self.LAND_TERRITORY_LOCKED
                land_block[lock_mask] = self.LAND_TERRITORY_UNLOCKED
                self.farm_grid[sx : sx + size, sy : sy + size] = land_block
                unlocked += 1
        return unlocked

    def _is_adjacent_harvested(self, agent_id: int, x: int, y: int) -> bool:
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx = x + dx
            ny = y + dy
            if nx < 0 or ny < 0 or nx >= self.config.width or ny >= self.config.height:
                continue
            if (
                int(self.owner_grid[nx, ny]) == agent_id
                and int(self.farm_grid[nx, ny]) == self.LAND_HARVESTED
            ):
                return True
        return False

    def _remaining_seed_budget(self) -> int:
        return max(0, self.config.total_seeds_per_episode - self.seeds_spawned)

    def check_episode_end(self) -> bool:
        for agent in self.agents:
            if agent.harvested_tiles >= int(getattr(self.config, "harvest_goal", 3)):
                self.winner = agent
                agent.reward += float(getattr(self.config, "win_reward", 50.0))
                self.done = True
                return True
        if self.steps >= self.config.max_steps:
            self.done = True
            return True
        if self._remaining_seed_budget() <= 0:
            self.done = True
            return True
        return False

    def _normalize_actions(
        self, actions: Dict[int, int] | Iterable[int] | int | None
    ) -> Dict[int, int]:
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


class FarmtillaVectorizedEnv(SimVectorEnv):
    """Lightweight wrapper that runs many FarmtilaEnv copies in parallel."""

    def __init__(self, config: FarmtilaConfig, num_envs: int | None = None) -> None:
        self.config = config
        resolved_envs = num_envs or getattr(config, "num_envs", 1)
        super().__init__(resolved_envs)

    def _create_env(self, index: int) -> SimEnv:
        return FarmtilaEnv(deepcopy(self.config))

    def _stack_rewards(self, reward_dicts: List[Dict[int, float]]) -> np.ndarray:
        reward_array = np.zeros((self.num_envs, self.config.num_agents), dtype=np.float32)
        for env_idx, rewards in enumerate(reward_dicts):
            for agent_id in range(self.config.num_agents):
                reward_array[env_idx, agent_id] = float(rewards.get(agent_id, 0.0))
        return reward_array

    def _stack_observations(self, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        obs_tensor = np.stack([obs["obs"] for obs in observations], axis=0)
        steps = np.array([obs.get("steps", 0) for obs in observations], dtype=np.int32)
        done_flags = np.array([obs.get("done", False) for obs in observations], dtype=np.bool_)
        return {
            "obs": obs_tensor,
            "agents": [obs.get("agents", []) for obs in observations],
            "done": done_flags,
            "winner": [obs.get("winner") for obs in observations],
            "steps": steps,
        }

    def assign_agents(self, agents: List[FarmtilaAgent]) -> None:
        """Assign distinct agent instances per environment.

        Vectorized environments should not share agent state (inventory, positions,
        harvested tiles) across envs. We clone agents so each sub-env can maintain
        its own counters while sharing policy objects.
        """
        self.agents = agents
        templates = {agent.agent_id: agent for agent in agents}
        for env in self.envs:
            positions = env._sample_unique_positions(env.config.num_agents)
            env_agents: List[FarmtilaAgent] = []
            for agent_id, (pos_x, pos_y) in enumerate(positions):
                template = templates.get(agent_id)
                policy = template.policy if template is not None else None
                action_space = getattr(template, "action_space", None)
                if not isinstance(action_space, np.ndarray):
                    default_action_count = getattr(
                        env.action_space, "n", FarmtilaEnv.ACTION_SPACE.n
                    )
                    action_count = getattr(action_space, "n", default_action_count)
                    action_space = np.arange(int(action_count), dtype=np.int64)
                env_agents.append(
                    FarmtilaAgent(
                        agent_id=agent_id,
                        position=(pos_x, pos_y),
                        action_space=action_space,
                        policy=policy,
                    )
                )
            env.assign_agents(env_agents)
