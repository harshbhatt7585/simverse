from abc import ABC, abstractmethod

import torch
from torch.nn import Module


class Policy(ABC, Module):
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        pass
