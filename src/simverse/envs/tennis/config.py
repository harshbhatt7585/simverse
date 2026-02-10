from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class TennisConfig:
    num_agents: int = 2
    num_envs: int = 1
    max_steps: int = 100_000
    seed: Optional[int] = None
    frame_skip: int = 4
    obs_resize: int = 84
    use_grayscale: bool = False
    policies: List[Any] = field(default_factory=list)
