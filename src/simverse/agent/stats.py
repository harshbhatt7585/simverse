from __future__ import annotations

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
    experiences: List[Experience]
    steps: int = 0


    def push_experience(self, experience: Experience) -> None:
        self.experiences.append(experience)
    

    def step(self) -> None:
        self.steps += 1


    def log_wandb(self, step: Optional[int] = None) -> None:
        if not _WANDB_AVAILABLE:
            return
        wandb.log(self.experiences, step=step)
