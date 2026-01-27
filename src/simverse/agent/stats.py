from __future__ import annotations

from pathlib import Path
import sys
if __package__ is None or __package__.startswith("__main__"):
    _src = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_src))

from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Optional
from simverse.utils.replay_buffer import Episode, Experience

try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    wandb = None
    _WANDB_AVAILABLE = False




@dataclass
class TrainingStats:
    experiences: List[Experience] = field(default_factory=list)
    steps: int = 0


    def push_experience(self, experience: Experience) -> None:
        self.experiences.append(experience)
    

    def step(self) -> None:
        self.steps += 1


    def log_wandb(self, step: Optional[int] = None) -> None:
        if not _WANDB_AVAILABLE:
            return
        wandb.log(self.experiences, step=step)

    
if __name__ == "__main__":
    import wandb
    import time
    import numpy as np
    wandb.init(project="simverse", name="stats")
    stats = TrainingStats()
    total_steps = 100
    wait_time = 0.3
    for step in range(total_steps):
        time.sleep(wait_time)
        stats.step()
        stats.push_experience(Experience(observation=np.array([1, 2, 3]), action=np.array([4, 5, 6]), log_prob=np.array([7, 8, 9]), value=np.array([10, 11, 12]), reward=np.array([13, 14, 15]), done=False, info={}))
        stats.log_wandb(step=stats.steps)


