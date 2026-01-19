from abc import ABC, abstractmethod
import numpy as np
from torch.nn import Module
import torch

class Policy(ABC, Module):
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        pass

