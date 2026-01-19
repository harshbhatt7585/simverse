"""Thin wrapper around Gymnasium's CartPole for a consistent interface."""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
from gymnasium.core import ActType, ObsType


class CartPoleEnv:
    """Convenience CartPole environment with reset/step helpers."""

    id = "CartPole-v1"

    def __init__(self, render_mode: Optional[str] = None, seed: Optional[int] = None):
        self._env = gym.make(self.id, render_mode=render_mode)
        self._seed = seed
        if seed is not None:
            self._env.reset(seed=seed)

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def reset(self) -> ObsType:
        obs, _info = self._env.reset(seed=self._seed)
        return obs

    def step(self, action: ActType):
        return self._env.step(action)

    def close(self) -> None:
        self._env.close()


__all__ = ["CartPoleEnv"]
