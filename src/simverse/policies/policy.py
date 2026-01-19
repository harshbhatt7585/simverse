from abc import ABC, abstractmethod
import numpy as np

class Policy(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def act(self, obs: np.ndarray) -> np.ndarray:
        pass