"Abstract class for all losses."
from abc import ABC, abstractmethod

class Loss(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> float:
        pass

    