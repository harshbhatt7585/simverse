from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class FarmtilaConfig:
    width: int = 50
    height: int = 50
    num_agents: int = 5
    num_envs: int = 1
    spawn_seed_every: int = 100
    seeds_per_spawn: int = 10
    max_steps: int = 10000
    total_seeds_per_episode: int = 500
    seed_proximity_reward_per_step: float = 0.02
    step_cost: float = 0.005
    territory_block_size: int = 3
    harvest_goal: int = 3
    territory_claim_reward: float = 0.1
    territory_unlock_reward: float = 5.0
    harvest_on_unlocked_reward: float = 1.0
    win_reward: float = 50.0
    adjacent_territory_step_cost_waiver: bool = True
    policies: List[Any] = field(default_factory=list)
