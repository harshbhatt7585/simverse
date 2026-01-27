from __future__ import annotations

from pathlib import Path
import random
import sys

if __package__ is None or __package__.startswith("__main__"):
    _src = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_src))

from simverse.envs.farmtila.render import FarmtilaRender
from simverse.envs.farmtila.env import FarmtilaEnv
from simverse.envs.farmtila.config import FarmtilaConfig
from simverse.envs.farmtila.agent import FarmtilaAgent
from simverse.simulator import Simulator
from simverse.policies.simple import SimplePolicy
from simverse.abstractor.policy import Policy
from simverse.losses.ppo import PPOTrainer
import torch


def agent_factory(agent_id: int, policy: Policy, env: FarmtilaEnv) -> FarmtilaAgent:
    return FarmtilaAgent(
        agent_id=agent_id,
        position=(random.randint(0, env.config.width - 1), random.randint(0, env.config.height - 1)),
        action_space=env.action_space,
        policy=policy,
    )


def run():
    # Config
    width, height, num_agents = 30, 20, 4
    
    # Create environment
    config = FarmtilaConfig(
        width=width,
        height=height,
        num_agents=num_agents,
        max_steps=500,
        total_seeds_per_episode=500,
        spawn_seed_every=50,
        seeds_per_spawn=10,
    )
    env = FarmtilaEnv(config=config)
    
    # Create policy (will be loaded from checkpoint)
    policy = SimplePolicy(
        obs_space=env.observation_space,
        action_space=env.action_space,
    )
    
    # Dummy trainer (not used for inference, but required by Simulator)
    loss_trainer = PPOTrainer(
        optimizer=torch.optim.Adam(policy.parameters(), lr=0.001),
        episodes=1,
    )
    
    # Create renderer with external control (Simulator controls stepping)
    renderer = FarmtilaRender(
        width=width,
        height=height,
        cell_size=32,
        external_control=True,
    )
    
    # Create simulator
    simulator = Simulator(
        env=env,
        num_agents=num_agents,
        policies=[policy],
        loss_trainer=loss_trainer,
        agent_factory=agent_factory,
    )
    
    # Run inference with checkpoint
    simulator.run(
        checkpoint_path="checkpoints/ppo_checkpoint_99.pth",
        renderer=renderer,
    )


if __name__ == "__main__":
    run()