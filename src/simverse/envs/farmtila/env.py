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
    LAND_OWNED = 1

    @property
    def action_space(self):
        return self.ACTION_SPACE

    @property
    def observation_space(self):
        # 5 channels: seed_grid, owner_grid, farm_grid(binary), agent_grid, inventory_grid
        return gym.spaces.Box(
            low=-1,
            high=max(self.config.num_agents, self.LAND_OWNED, 1),
            shape=(5, self.config.width, self.config.height),
            dtype=np.float32,
        )

    def __init__(self, config: FarmtilaConfig):
        self.config = config
        if self.config.num_agents != 2:
            raise ValueError("Competitive Farmtila requires exactly 2 agents")

        self.seed_grid = np.zeros((config.width, config.height), dtype=np.uint8)
        self.owner_grid = np.full((config.width, config.height), -1, dtype=np.int16)
        self.farm_grid = np.zeros((config.width, config.height), dtype=np.uint8)

        self.agents: List[FarmtilaAgent] = []
        self.rng = np.random.default_rng()

        self.steps = 0
        self.last_pickups: List[Tuple[int, int, int]] = []
        self.seeds_spawned = 0
        self.done = False
        self.winner: FarmtilaAgent | None = None

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
        prev_score_delta = self._score_delta()
        step_cost = float(getattr(self.config, "step_cost", 0.0))

        for agent in self.agents:
            reward = -step_cost
            action = action_map.get(agent.agent_id)
            if action is not None:
                dx, dy = self._action_to_delta(action)
                new_x = int(np.clip(agent.position[0] + dx, 0, self.config.width - 1))
                new_y = int(np.clip(agent.position[1] + dy, 0, self.config.height - 1))
                agent.position = (new_x, new_y)

                if self._collect_seed_if_present(agent):
                    reward += 0.0

                if action == self.HARVEST_ACTION:
                    self._claim_or_steal(agent)

            agent.reward += reward

        self.steps += 1
        self._spawn_seeds_if_due()
        self.check_episode_end()

        score_delta_reward = float(getattr(self.config, "score_delta_reward", 1.0))
        delta_change = self._score_delta() - prev_score_delta
        if len(self.agents) == 2 and score_delta_reward != 0.0:
            self.agents[0].reward += delta_change * score_delta_reward
            self.agents[1].reward -= delta_change * score_delta_reward

        return self._package_step_result()

    def step_random(self):
        actions = {agent.agent_id: int(self.rng.integers(0, 4)) for agent in self.agents}
        return self.step(actions)

    def render(self):
        pass

    def get_observation(self):
        return self._get_observation()

    def assign_agents(self, agents: List[FarmtilaAgent]) -> None:
        if len(agents) != 2:
            raise ValueError("Competitive Farmtila requires exactly 2 agents")
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
        agent_grid = np.zeros((self.config.width, self.config.height), dtype=np.float32)
        inventory_grid = np.zeros((self.config.width, self.config.height), dtype=np.float32)
        for agent in self.agents:
            x, y = agent.position
            agent_grid[x, y] = agent.agent_id + 1
            inventory_grid[x, y] = agent.inventory

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

    def _claim_or_steal(self, agent: FarmtilaAgent) -> None:
        if agent.inventory <= 0:
            return
        x, y = agent.position
        owner = int(self.owner_grid[x, y])
        if owner == agent.agent_id:
            return

        agent.inventory -= 1
        self.owner_grid[x, y] = agent.agent_id
        self.farm_grid[x, y] = self.LAND_OWNED
        agent.harvested_tiles += 1

        if owner >= 0 and owner < len(self.agents):
            previous_owner = self.agents[owner]
            previous_owner.harvested_tiles = max(0, previous_owner.harvested_tiles - 1)

    def _remaining_seed_budget(self) -> int:
        return max(0, self.config.total_seeds_per_episode - self.seeds_spawned)

    def _score_delta(self) -> int:
        if len(self.agents) < 2:
            return 0
        return int(self.agents[0].harvested_tiles - self.agents[1].harvested_tiles)

    def _finalize_winner(self) -> None:
        if len(self.agents) != 2:
            self.winner = None
            return
        if self.agents[0].harvested_tiles > self.agents[1].harvested_tiles:
            self.winner = self.agents[0]
        elif self.agents[1].harvested_tiles > self.agents[0].harvested_tiles:
            self.winner = self.agents[1]
        else:
            self.winner = None

    def check_episode_end(self) -> bool:
        seeds_exhausted = self._remaining_seed_budget() <= 0 and np.sum(self.seed_grid) == 0
        no_inventory = all(agent.inventory <= 0 for agent in self.agents)

        if self.steps >= self.config.max_steps or (seeds_exhausted and no_inventory):
            self.done = True
            self._finalize_winner()
            if self.winner is not None:
                terminal_reward = float(getattr(self.config, "terminal_win_reward", 1.0))
                winner_id = self.winner.agent_id
                loser_id = 1 - winner_id
                self.agents[winner_id].reward += terminal_reward
                self.agents[loser_id].reward -= terminal_reward
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
        if len(agents) != 2:
            raise ValueError("Competitive Farmtila requires exactly 2 agents")
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
