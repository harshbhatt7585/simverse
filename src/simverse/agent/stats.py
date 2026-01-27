from __future__ import annotations

from pathlib import Path
import sys
if __package__ is None or __package__.startswith("__main__"):
    _src = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_src))

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from simverse.utils.replay_buffer import Experience
import numpy as np

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
    episode_count: int = 0
    step_rewards: List[float] = field(default_factory=list)  # Per-step rewards
    episode_rewards: List[float] = field(default_factory=list)  # Total episode rewards
    policy_losses: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)

    def push_experience(self, experience: Experience) -> None:
        self.experiences.append(experience)
        # Track per-step reward
        reward = experience.reward
        if isinstance(reward, dict):
            reward = sum(reward.values())
        self.step_rewards.append(float(reward) if reward else 0.0)

    def push_losses(self, policy_loss: float, value_loss: float) -> None:
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)

    def push_reward(self, reward: float) -> None:
        """Push total episode reward."""
        self.episode_rewards.append(reward)
        self.episode_count += 1

    def step(self) -> None:
        self.steps += 1

    def log_wandb(self, step: Optional[int] = None) -> None:
        if not _WANDB_AVAILABLE:
            return
        payload = {}
        payload["trainer/steps"] = self.steps
        payload["trainer/episodes"] = self.episode_count
    
        # Step-level rewards
        if self.step_rewards:
            payload["step/reward"] = self.step_rewards[-1]
            
        # Episode-level rewards
        if self.episode_rewards:
            payload["episode/reward"] = self.episode_rewards[-1]
            payload["episode/cumulative_reward"] = sum(self.episode_rewards)
            payload["episode/avg_reward"] = sum(self.episode_rewards) / len(self.episode_rewards)
        
        if self.experiences:
            last = self.experiences[-1]
            payload["episode/done"] = np.float32(last.done).item()
            
        if self.policy_losses:
            payload["loss/policy"] = self.policy_losses[-1]
            payload["loss/policy_avg"] = sum(self.policy_losses) / len(self.policy_losses)
            
        if self.value_losses:
            payload["loss/value"] = self.value_losses[-1]
            payload["loss/value_avg"] = sum(self.value_losses) / len(self.value_losses)
            
        wandb.log(payload, step=step)
    
    def reset_episode(self) -> None:
        """Reset episode-level stats."""
        self.step_rewards.clear()
        self.experiences.clear()

    
if __name__ == "__main__":
    import wandb
    import time
    import numpy as np
    wandb.init(project="simverse", name="stats-test")
    stats = TrainingStats()
    total_steps = 100
    wait_time = 0.3
    for step in range(total_steps):
        time.sleep(wait_time)
        stats.step()
        stats.push_experience(Experience(
            observation=np.array([1, 2, 3]), 
            action=np.array([4, 5, 6]), 
            log_prob=np.array([7, 8, 9]), 
            value=np.array([10, 11, 12]), 
            reward=0.5,  # Simple float reward
            done=False, 
            info={}
        ))
        stats.push_losses(policy_loss=0.1 * step, value_loss=0.05 * step)
        stats.log_wandb(step=stats.steps)
    wandb.finish()
