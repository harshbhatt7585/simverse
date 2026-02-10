from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class FarmtilaConfig:
    width: int = 50
    height: int = 50
    num_agents: int = 2
    num_envs: int = 1
    spawn_seed_every: int = 100
    seeds_per_spawn: int = 10
    max_steps: int = 10000
    total_seeds_per_episode: int = 500
    step_cost: float = 0.0
    score_delta_reward: float = 1.0
    terminal_win_reward: float = 1.0
    policies: List[Any] = field(default_factory=list)
