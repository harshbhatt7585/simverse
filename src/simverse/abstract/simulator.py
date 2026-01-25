from abc import ABC, abstractmethod

class Simulator(ABC):
    @abstractmethod
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def run(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        pass


    