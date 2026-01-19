from abc import ABC, abstractmethod
import numpy as np
from typing import Optional
from simverse.policies.policy import Policy

class SimAgent(ABC):
    @abstractmethod
    def __init__(
        self,
        name: str,
        action_space: np.ndarray,
        policy: Optional[Policy] = None,
        **kwargs,
    ) -> None:
        self.name = name
        self.action_space = action_space
        self.policy = policy if policy is not None else None

    @abstractmethod
    def action(self, obs: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def info(self) -> dict:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def get_action_space(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_memory(self) -> dict:
        pass
    
    @abstractmethod
    def current_state(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_policy(self) -> Policy:
        pass

    @abstractmethod
    def set_policy(self, policy: Policy) -> None:
        pass
    
   