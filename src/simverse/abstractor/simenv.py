from abc import ABC, abstractmethod

class SimEnv(ABC):
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
