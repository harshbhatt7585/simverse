from __future__ import annotations

from pathlib import Path
import random
import sys

if __package__ is None or __package__.startswith("__main__"):
    _src = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_src))

from simverse.simulator import Simulator

from simverse.envs.farmtila.env import FarmtilaEnv
from simverse.envs.farmtila.config import FarmtilaConfig

from simverse.config.policy import PolicySpec
from simverse.policies.simple import SimplePolicy
from simverse.losses.ppo import PPOTrainer
import torch
from simverse.abstractor.policy import Policy
from simverse.envs.farmtila.agent import FarmtilaAgent


def agent_factory(agent_id: int, policy: Policy, env: FarmtilaEnv) -> FarmtilaAgent:
    return FarmtilaAgent(
        agent_id=agent_id,
        position=(random.randint(0, env.config.width - 1), random.randint(0, env.config.height - 1)),
        action_space=env.action_space,
        policy=policy,
    )


def train():
    config = FarmtilaConfig(
        width=30,
        height=20,
        num_agents=4,
        total_seeds_per_episode=500,
        max_steps=10000,
        spawn_seed_every=100,
        seeds_per_spawn=10,
        policies=[],
    )
    env = FarmtilaEnv(config=config)
    policy_spec = PolicySpec(
        name="simple",
        model=SimplePolicy(
            obs_space=env.observation_space,
            action_space=env.action_space,
        ),
    )
    env.config.policies = [policy_spec]

    loss_trainer = PPOTrainer(
        optimizer=torch.optim.Adam(env.policy.parameters(), lr=0.001),
        episodes=100,
        training_epochs=10,
        clip_epsilon=0.2,
    )
    simulator = Simulator(
        env=env,
        num_agents=4,
        policies=env.config.policies,
        loss_trainer=loss_trainer,
        agent_factory=agent_factory
    )

    # starts the training
    simulator.train()



if __name__ == "__main__":
    train()
    
