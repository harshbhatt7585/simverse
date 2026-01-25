from simverse.policies.policy import Policy
import torch.nn as nn
import torch
from torch.nn import functional as F

class SimplePolicy(Policy):
    def __init__(
        self,
        obs_space ,
        action_space,
    ):
        super().__init__()


        # obs encoder
        self.obs_encoder = nn.Linear(obs_space.shape[0], 128)

        # decoder
        self.fc1 = nn.Linear(128, 128)
        
        # action head
        self.action_head = nn.Linear(128, action_space.n)

        # value head
        self.value_head = nn.Linear(128, 1)

    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:

        
        x = self.obs_encoder(obs)
        x = F.relu(self.fc1(x))

        logits = self.action_head(x)

        value = self.value_head(x)

        return logits, value
        

    
