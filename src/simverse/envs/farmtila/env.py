from simverse.envs.farmtila.config import FarmtilaConfig
import numpy as np
from simverse.agent.sim_agent import SimAgent
from typing import List

class FarmtilaEnv():
    def __init__(self, config: FarmtilaConfig):
        self.config = config

        self.seed_grid = np.zeros((config.width, config.height))
        self.owner_grid = np.full((config.width, config.height), -1)

        # Agent class have their own position

        self.agents: List[SimAgent] = []

        self.steps = 0
    
    def reset(self):
        pass
    
    def step(self):
        pass

    def render(self):
        pass

