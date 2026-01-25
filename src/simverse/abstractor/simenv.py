from abc import ABC, abstractmethod
import gymnasium as gym

class SimEnv(ABC):


    @property
    @abstractmethod
    def action_space(self) -> gym.spaces.Space:
        pass

    @abstractmethod
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def step(self, action: int) -> None:
        pass

    @abstractmethod
    def get_observation(self) -> None:
        pass
