from dataclasses import dataclass

import torch.nn as nn


@dataclass
class PolicySpec:
    name: str
    model: nn.Module
