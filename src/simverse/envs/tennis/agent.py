from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from simverse.abstractor.agent import SimAgent

if TYPE_CHECKING:
    from simverse.abstractor.policy import Policy


class TennisAgent(SimAgent):
    def __init__(
        self,
        agent_id: int,
        action_space: np.ndarray,
        policy: Optional["Policy"] = None,
        name: str | None = None,
    ) -> None:
        super().__init__(
            name=name or f"tennis_agent_{agent_id}",
            action_space=action_space,
            policy=policy,
        )
        self.agent_id = agent_id
        self.reward = 0.0
        self.memory: dict = {}
        self._rng = np.random.default_rng(agent_id)

    def action(self, obs: np.ndarray) -> np.ndarray:
        if self.policy is not None:
            return self.policy(obs)
        return np.array([self._rng.choice(self.action_space)], dtype=np.int64)

    def info(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "reward": self.reward,
            "name": self.name,
        }

    def reset(self) -> None:
        self.reward = 0.0
        self.memory.clear()

    def get_action_space(self) -> np.ndarray:
        return self.action_space

    def get_memory(self) -> dict:
        return self.memory

    def current_state(self) -> np.ndarray:
        return np.array([self.reward], dtype=np.float32)

    def get_policy(self):
        return self.policy

    def set_policy(self, policy) -> None:
        self.policy = policy
