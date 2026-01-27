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
from simverse.agent.stats import TrainingStats
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
    # Training hyperparameters
    training_config = {
        "width": 30,
        "height": 20,
        "num_agents": 4,
        "max_steps": 100,
        "episodes": 100,
        "training_epochs": 10,
        "lr": 0.001,
        "clip_epsilon": 0.2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "total_seeds": 500,
    }
    
    config = FarmtilaConfig(
        width=training_config["width"],
        height=training_config["height"],
        num_agents=training_config["num_agents"],
        total_seeds_per_episode=training_config["total_seeds"],
        max_steps=training_config["max_steps"],
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
    
    policy_models = [ps.model for ps in env.config.policies]
    
    # Create stats tracker
    stats = TrainingStats()

    # Create trainer with config for logging
    loss_trainer = PPOTrainer(
        optimizer=torch.optim.Adam(policy_spec.model.parameters(), lr=training_config["lr"]),
        episodes=training_config["episodes"],
        training_epochs=training_config["training_epochs"],
        clip_epsilon=training_config["clip_epsilon"],
        gamma=training_config["gamma"],
        gae_lambda=training_config["gae_lambda"],
        stats=stats,
        config=training_config,
        project_name="simverse-farmtila",
        run_name="ppo-training",
    )
    
    simulator = Simulator(
        env=env,
        num_agents=training_config["num_agents"],
        policies=policy_models,
        loss_trainer=loss_trainer,
        agent_factory=agent_factory
    )

    # Start training
    simulator.train(title="Farmtila Training")



if __name__ == "__main__":
    train()
    
