from dataclasses import dataclass
import numpy as np

@dataclass
class FarmtilaConfig:
    width: int = 50
    height: int = 50
    num_agents: int = 5
    spawn_seed_every: int = 100
    seeds_per_spawn: int = 10
    max_steps: int = 10000
    total_seeds_per_episode: int = 500
