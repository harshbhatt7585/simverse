from abc import ABC, abstractmethod
from typing import List
from simverse.abstractor.simenv import SimEnv
from simverse.abstractor.policy import Policy
from simverse.abstractor.trainer import Trainer
import random
import torch

class Simulator(ABC):
    def __init__(self, env: SimEnv, num_agents: int, policies: List[Policy], loss_trainer: Trainer) -> None:
        self.env = env
        self.num_agents = num_agents
        self.policies = policies
        self.loss_trainer = loss_trainer

    def run(self) -> None:
        agents = []
        for _ in range(self.num_agents):


            agent = SimAgent(
                name=f"agent_{i}", 
                action_space=self.env.action_space, 
                policy=random.choice(self.policies).name
            )
            agents.append(agent)
        

    def train(
        self,
        *args,
        **kwargs,
    ) -> None:

        agents = []
        for _ in range(self.num_agents):


            agent = SimAgent(
                name=f"agent_{i}", 
                action_space=self.env.action_space, 
                policy=random.choice(self.policies).name
            )
            agents.append(agent)
        

        self.loss_trainer.train(
            env=self.env,
            agents=agents,
            *args,
            **kwargs,
        )
        



    