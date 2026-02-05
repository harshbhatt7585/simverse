from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Sequence

import numpy as np

from simverse.abstractor.simenv import SimEnv


class SimVectorEnv(SimEnv, ABC):
    """Base class that handles running multiple SimEnv instances in lockstep."""

    def __init__(self, num_envs: int) -> None:
        if num_envs <= 0:
            raise ValueError("SimVectorEnv requires at least one environment instance")
        self.num_envs = num_envs
        self.envs: List[SimEnv] = [self._create_env(idx) for idx in range(num_envs)]
        self._last_obs: List[Any] = []
        self.steps = 0
        self.agents: List[Any] = []

    @abstractmethod
    def _create_env(self, index: int) -> SimEnv:
        """Return a freshly constructed sub-environment."""

    def reset(self):  # type: ignore[override]
        self.steps = 0
        self._last_obs = [env.reset() for env in self.envs]
        return self._stack_observations(self._last_obs)

    def step(self, actions: Sequence[Any]):  # type: ignore[override]
        if len(actions) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} action collections, received {len(actions)} instead"
            )

        obs_batch: List[Any] = []
        reward_batch: List[Any] = []
        done_batch: List[Any] = []
        info_batch: List[Any] = []

        for env, env_actions in zip(self.envs, actions):
            obs, reward, done, info = env.step(env_actions)
            obs_batch.append(obs)
            reward_batch.append(reward)
            done_batch.append(done)
            info_batch.append(info)

        self._last_obs = obs_batch
        self.steps += 1
        stacked_obs = self._stack_observations(obs_batch)
        stacked_rewards = self._stack_rewards(reward_batch)
        stacked_dones = self._stack_dones(done_batch)
        stacked_info = self._stack_info(info_batch)
        return stacked_obs, stacked_rewards, stacked_dones, stacked_info

    def get_observation(self):  # type: ignore[override]
        if not self._last_obs:
            return self.reset()
        return self._stack_observations(self._last_obs)

    @property
    def action_space(self):  # type: ignore[override]
        return self.envs[0].action_space

    @property
    def observation_space(self):  # type: ignore[override]
        return self.envs[0].observation_space

    def assign_agents(self, agents: List[Any]) -> None:
        self.agents = agents
        for env in self.envs:
            assign = getattr(env, "assign_agents", None)
            if callable(assign):
                assign(agents)

    @abstractmethod
    def _stack_observations(self, observations: List[Any]) -> Any:
        """Convert raw per-env observations into a batch structure."""

    @abstractmethod
    def _stack_rewards(self, rewards: List[Any]) -> Any:
        """Convert raw per-env rewards into a batch structure."""

    def _stack_dones(self, dones: List[Any]) -> np.ndarray:
        return np.asarray(dones, dtype=np.bool_)

    def _stack_info(self, info: List[Any]) -> List[Any]:
        return info
