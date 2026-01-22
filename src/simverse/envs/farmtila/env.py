from __future__ import annotations

from dataclasses import dataclass
from typing import List

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

        self.steps = 0
    
    def reset(self):
        self.seed_grid.fill(0)
        self.owner_grid.fill(-1)
        self.agents = self._spawn_agents()
        self.steps = 0
        return self._get_observation()
    
    def step(self):
        pass

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
