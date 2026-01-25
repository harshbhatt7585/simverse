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
        obs_space,
        action_space,
    ):
        super().__init__()

        channels, height, width = obs_space.shape
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            conv_out_dim = self.obs_encoder(dummy).shape[1]

        self.fc1 = nn.Linear(conv_out_dim, 128)
        
        # action head
        self.action_head = nn.Linear(128, action_space.n)

        # value head
        self.value_head = nn.Linear(128, 1)

    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        x = self.obs_encoder(obs)
        x = F.relu(self.fc1(x))

        logits = self.action_head(x)

        value = self.value_head(x)

        return logits, value
        


if __name__ == "__main__":
    import gymnasium as gym
    
    obs_space = gym.spaces.Box(0, 1, shape=(3, 30, 20))
    action_space = gym.spaces.Discrete(6)
    policy = SimplePolicy(obs_space, action_space)
    obs = torch.randn(1, *obs_space.shape)
    logits, value = policy(obs)
    print(logits)
    print(value)

    
