from abc import ABC, abstractmethod

class Trainer(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        pass