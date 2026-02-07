from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class SimTorchEnv(nn.Module, ABC):
    """Base class for torch-native, batched simulation environments."""

    def __init__(self, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype

    @property
    @abstractmethod
    def action_space(self) -> Any: ...

    @property
    @abstractmethod
    def observation_space(self) -> Any: ...

    @abstractmethod
    def reset(self) -> Dict[str, torch.Tensor]: ...

    @abstractmethod
    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]: ...

    def get_observation(self) -> Dict[str, torch.Tensor]:
        return self.reset()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if args:
            self.device = torch.device(args[0])
        elif "device" in kwargs:
            self.device = torch.device(kwargs["device"])
        return self
