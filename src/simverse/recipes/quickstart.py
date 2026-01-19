"""Utility function to run a short random-policy rollout."""

from __future__ import annotations

from typing import Callable, Dict, List, Type

import numpy as np

from simverse.envs import CartPoleEnv
from simverse.policies import RandomPolicy


def quicktrain(
    episodes: int = 5,
    max_steps: int = 200,
    render: bool = False,
    env_cls: Type[CartPoleEnv] = CartPoleEnv,
    policy_cls: Type[RandomPolicy] = RandomPolicy,
) -> Dict[str, float]:
    """Runs a basic training loop using the provided env/policy."""

    render_mode = "human" if render else None
    env = env_cls(render_mode=render_mode)
    policy = policy_cls(env.action_space)

    rewards: List[float] = []
    for episode in range(episodes):
        obs = env.reset()
        total_reward = 0.0
        for _step in range(max_steps):
            action = policy.act(obs)
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)

    env.close()

    stats = {
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "max_reward": float(np.max(rewards)) if rewards else 0.0,
    }
    return stats


__all__ = ["quicktrain"]
