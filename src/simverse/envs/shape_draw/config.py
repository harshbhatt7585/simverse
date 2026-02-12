from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class ShapeDrawConfig:
    width: int = 64
    height: int = 64
    num_agents: int = 1
    num_envs: int = 64
    max_steps: int = 256
    seed: Optional[int] = None
    min_brush: int = 1
    max_brush: int = 5
    draw_penalty: float = 0.001
    step_penalty: float = 0.0
    completion_bonus: float = 2.0
    completion_threshold: float = 0.95
    policies: List[Any] = field(default_factory=list)
