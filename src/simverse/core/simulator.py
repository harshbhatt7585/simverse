from __future__ import annotations

import random
from typing import Callable, List, Optional, Protocol

import torch
import torch.nn as nn
from torch.distributions import Categorical

from simverse.core.agent import SimAgent
from simverse.core.env import SimEnv
from simverse.core.trainer import Trainer
from simverse.utils.checkpointer import Checkpointer

AgentFactory = Callable[[int, nn.Module, SimEnv], SimAgent]


class Renderer(Protocol):
    """Protocol for environment renderers."""

    def draw(self, env: SimEnv) -> None: ...
    def handle_events(self) -> None: ...
    def close(self) -> None: ...


class Simulator:
    """High-level orchestrator that spawns agents and delegates training."""

    def __init__(
        self,
        env: SimEnv,
        num_agents: int,
        policies: List[nn.Module],
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

        self.checkpointer = Checkpointer(self.env)

    def _build_agents(self) -> List[SimAgent]:
        agents: List[SimAgent] = []
        assign_by_index = len(self.policies) == self.num_agents
        single_policy = len(self.policies) == 1
        for idx in range(self.num_agents):
            if assign_by_index:
                policy = self.policies[idx]
            elif single_policy:
                policy = self.policies[0]
            else:
                policy = random.choice(self.policies)
            agent = self.agent_factory(idx, policy, self.env)
            agents.append(agent)
        return agents

    def train(self, *args, **kwargs) -> None:
        agents = self._build_agents()
        if hasattr(self.env, "assign_agents") and callable(self.env.assign_agents):
            self.env.assign_agents(agents)
        elif hasattr(self.env, "agents"):
            self.env.agents = agents
        self.loss_trainer.train(self.env, agents, *args, **kwargs)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        self.checkpointer.load(checkpoint_path)

    def run(
        self,
        checkpoint_path: Optional[str] = None,
        max_steps: Optional[int] = None,
        renderer: Optional[Renderer] = None,
    ) -> None:
        """
        Run inference with trained policies.

        Args:
            checkpoint_path: Path to load model checkpoint from
            max_steps: Maximum steps to run (defaults to env.config.max_steps)
            renderer: Optional renderer instance for visualization
        """
        agents = self._build_agents()
        if hasattr(self.env, "assign_agents") and callable(self.env.assign_agents):
            self.env.assign_agents(agents)
        elif hasattr(self.env, "agents"):
            self.env.agents = agents

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        obs = self.env.reset()
        max_steps = max_steps or getattr(getattr(self.env, "config", None), "max_steps", None)
        done = False
        step = 0

        try:
            while not done and (max_steps is None or step < max_steps):
                if renderer:
                    renderer.handle_events()

                actions = {}
                for agent in agents:
                    if agent.policy is None:
                        continue
                    agent.policy.eval()
                    with torch.no_grad():
                        obs_tensor = torch.from_numpy(obs["obs"]).float().unsqueeze(0)
                        logits, _ = agent.policy(obs_tensor)
                        action = Categorical(logits=logits).sample().item()
                    actions[agent.agent_id] = action

                obs, reward, done, info = self.env.step(actions)

                if renderer:
                    renderer.draw(self.env)

                step += 1
        finally:
            if renderer:
                renderer.close()
