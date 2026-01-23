from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from .config import FarmtilaConfig


@dataclass
class FarmtilaAgent:
    agent_id: int
    position: tuple[int, int]
    inventory: int = 0

class FarmtilaEnv():
    def __init__(self, config: FarmtilaConfig):
        self.config = config

        self.seed_grid = np.zeros((config.width, config.height))
        self.owner_grid = np.full((config.width, config.height), -1)

        # Agent class have their own position

        self.agents: List[FarmtilaAgent] = []

        self.rng = np.random.default_rng()

        self.steps = 0
    
    def reset(self):
        self.seed_grid.fill(0)
        self.owner_grid.fill(-1)
        self.agents = self._spawn_agents()
        self.steps = 0
        return self._get_observation()
    
    def step(self, actions: Dict[int, int] | Iterable[int] | int | None = None):
        """Advance the simulation by applying actions to agents."""
        action_map = self._normalize_actions(actions)
        for agent in self.agents:
            action = action_map.get(agent.agent_id)
            if action is None:
                continue
            dx, dy = self._action_to_delta(action)
            if dx == 0 and dy == 0:
                continue
            new_x = int(np.clip(agent.position[0] + dx, 0, self.config.width - 1))
            new_y = int(np.clip(agent.position[1] + dy, 0, self.config.height - 1))
            agent.position = (new_x, new_y)
        self.steps += 1
        return self._get_observation()

    def step_random(self):
        """Apply a random move to every agent."""
        actions = {
            agent.agent_id: int(self.rng.integers(0, 4))  # movement actions only
            for agent in self.agents
        }
        return self.step(actions)

    def render(self):
        pass

    def _spawn_agents(self) -> List[FarmtilaAgent]:
        agents: List[FarmtilaAgent] = []
        occupied = set()
        rng = np.random.default_rng()
        for agent_id in range(self.config.num_agents):
            while True:
                x = int(rng.integers(0, self.config.width))
                y = int(rng.integers(0, self.config.height))
                if (x, y) not in occupied:
                    occupied.add((x, y))
                    break
            agents.append(FarmtilaAgent(agent_id=agent_id, position=(x, y)))
        return agents

    def _get_observation(self):
        return {
            "seed_grid": self.seed_grid.copy(),
            "owner_grid": self.owner_grid.copy(),
            "agents": [agent.position for agent in self.agents],
        }

    def _normalize_actions(self, actions: Dict[int, int] | Iterable[int] | int | None) -> Dict[int, int]:
        if actions is None:
            return {}
        if isinstance(actions, dict):
            return actions
        if isinstance(actions, int):
            return {0: actions}
        # treat iterable as ordered by agent id
        return {agent_id: action for agent_id, action in enumerate(actions)}

    def _action_to_delta(self, action: int) -> tuple[int, int]:
        return {
            0: (0, -1),  # up
            1: (0, 1),   # down
            2: (-1, 0),  # left
            3: (1, 0),   # right
        }.get(action, (0, 0))
