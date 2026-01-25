from pathlib import Path
import sys

if __name__ == "__main__" and __package__ is None:
    _src = Path(__file__).resolve().parents[2]  # path/to/src
    sys.path.insert(0, str(_src))


import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from simverse.policies.policy import Policy



class SimplePolicy(Policy):
    def __init__(
        self,
        obs_space ,
        action_space,
    ):
        super().__init__()


        # obs encoder
        self.obs_dim = int(np.prod(obs_space.shape))
        self.obs_encoder = nn.Linear(self.obs_dim, 128)

        # decoder
        self.fc1 = nn.Linear(128, 128)
        
        # action head
        self.action_head = nn.Linear(128, action_space.n)

        # value head
        self.value_head = nn.Linear(128, 1)

    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.view(obs.size(0), -1)
        x = self.obs_encoder(x)
        x = F.relu(self.fc1(x))

        logits = self.action_head(x)

        value = self.value_head(x)

        return logits, value
        


if __name__ == "__main__":
    import gymnasium as gym

    obs_space = gym.spaces.Box(0, 1, shape=(10, 10, 3))
    action_space = gym.spaces.Discrete(6)
    policy = SimplePolicy(obs_space, action_space)
    obs = torch.randn(1, 10, 10, 3)
    logits, value = policy(obs)
    print(logits)
    print(value)

    
