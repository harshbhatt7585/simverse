from __future__ import annotations

import random
from typing import Callable, List

from simverse.abstractor.agent import SimAgent
from simverse.abstractor.policy import Policy
from simverse.abstractor.simenv import SimEnv
from simverse.abstractor.trainer import Trainer


AgentFactory = Callable[[int, Policy, SimEnv], SimAgent]


class Simulator:
    """High-level orchestrator that spawns agents and delegates training."""

    def __init__(
        self,
        env: SimEnv,
        num_agents: int,
        policies: List[Policy],
        loss_trainer: Trainer,
        agent_factory: AgentFactory,
    ) -> None:
        if not policies:
            raise ValueError("Simulator requires at least one policy instance")
        self.env = env
        self.num_agents = num_agents
        self.policies = policies
        self.loss_trainer = loss_trainer
        self.agent_factory = agent_factory

    def _build_agents(self) -> List[SimAgent]:
        agents: List[SimAgent] = []
        for idx in range(self.num_agents):
            policy = random.choice(self.policies)
            agent = self.agent_factory(idx, policy, self.env)
            agents.append(agent)
        return agents

    def run(self, *args, **kwargs) -> None:
        agents = self._build_agents()
        self.loss_trainer.train(self.env, agents, *args, **kwargs)

    # Alias for readability
    train = run



    
