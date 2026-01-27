from abc import ABC, abstractmethod
from simverse.utils.checkpointer import Checkpointer

class Trainer(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        pass


    def save_checkpoint(self, checkpoint_path: str) -> None:
        checkpointer = Checkpointer(self.env)
        checkpointer.save(checkpoint_path)
