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
from simverse.logging_config import training_logger
import torch
from simverse.abstractor.policy import Policy
from simverse.envs.farmtila.agent import FarmtilaAgent

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


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
    
    # Beautiful header
    training_logger.header("Farmtila Training")
    training_logger.config(training_config)
    
    # Initialize wandb for logging
    if WANDB_AVAILABLE:
        training_logger.info("Weights & Biases logging enabled")
        wandb.init(
            project="simverse-farmtila",
            name="ppo-training",
            config=training_config
        )
    else:
        training_logger.warning("Weights & Biases not available - install with: pip install wandb")
    
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

    loss_trainer = PPOTrainer(
        optimizer=torch.optim.Adam(policy_spec.model.parameters(), lr=training_config["lr"]),
        episodes=training_config["episodes"],
        training_epochs=training_config["training_epochs"],
        clip_epsilon=training_config["clip_epsilon"],
        gamma=training_config["gamma"],
        gae_lambda=training_config["gae_lambda"],
        stats=stats,
    )
    simulator = Simulator(
        env=env,
        num_agents=training_config["num_agents"],
        policies=policy_models,  # Pass the actual models, not PolicySpec
        loss_trainer=loss_trainer,
        agent_factory=agent_factory
    )

    # starts the training
    training_logger.success("Environment and policies initialized")
    simulator.train()
    
    # Finish wandb run
    if WANDB_AVAILABLE:
        wandb.finish()
        training_logger.success("Wandb run finished")



if __name__ == "__main__":
    train()
    
