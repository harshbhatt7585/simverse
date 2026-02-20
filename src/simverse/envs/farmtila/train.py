from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

if __package__ is None or __package__.startswith("__main__"):
    _src = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_src))

import torch

from simverse.abstractor.policy import Policy
from simverse.abstractor.train_utils import build_adam_optimizers
from simverse.agent.stats import TrainingStats
from simverse.config.policy import PolicySpec
from simverse.envs.farmtila.agent import FarmtilaAgent
from simverse.envs.farmtila.config import FarmtilaConfig
from simverse.envs.farmtila.env import FarmtilaEnv, create_env
from simverse.envs.farmtila.training_config import build_training_config
from simverse.losses.ppo import PPOTrainer
from simverse.policies.simple import SimplePolicy
from simverse.simulator import Simulator
from simverse.wandb_config import DEFAULT_WANDB_PROJECT


def agent_factory(agent_id: int, policy: Policy, env: FarmtilaEnv) -> FarmtilaAgent:
    return FarmtilaAgent(
        agent_id=agent_id,
        position=(
            random.randint(0, env.config.width - 1),
            random.randint(0, env.config.height - 1),
        ),
        action_space=env.action_space,
        policy=policy,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Farmtila PPO agents")
    parser.add_argument(
        "--wandb",
        choices=["on", "off"],
        default="on",
        help="Enable or disable Weights & Biases logging",
    )
    return parser.parse_args()


def train(use_wandb: bool = True):
    training_config = build_training_config(
        num_agents=2,
        num_envs=2048,
        max_steps=1500,
        episodes=100,
        training_epochs=1,
        lr=0.001,
        clip_epsilon=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        total_seeds=500,
        batch_size=None,
        buffer_size=None,
        dtype=torch.float16,
    )

    config = FarmtilaConfig(
        width=training_config["width"],
        height=training_config["height"],
        num_agents=training_config["num_agents"],
        num_envs=training_config["num_envs"],
        total_seeds_per_episode=training_config["total_seeds"],
        max_steps=training_config["max_steps"],
        spawn_seed_every=100,
        seeds_per_spawn=10,
        policies=[],
    )
    env = create_env(
        config=config,
        num_envs=training_config["num_envs"],
        device=training_config["device"],
        dtype=training_config["dtype"],
    )
    policy_specs = [
        PolicySpec(
            name=f"simple_agent_{agent_id}",
            model=SimplePolicy(
                obs_space=env.observation_space,
                action_space=env.action_space,
            ),
        )
        for agent_id in range(training_config["num_agents"])
    ]
    env.config.policies = policy_specs

    policy_models = [ps.model for ps in env.config.policies]
    optimizers = build_adam_optimizers(
        policy_models,
        lr=training_config["lr"],
        device=training_config["device"],
    )

    # Create stats tracker
    stats = TrainingStats()

    # Create trainer with config for logging
    loss_trainer = PPOTrainer(
        optimizers=optimizers,
        episodes=training_config["episodes"],
        training_epochs=training_config["training_epochs"],
        clip_epsilon=training_config["clip_epsilon"],
        gamma=training_config["gamma"],
        gae_lambda=training_config["gae_lambda"],
        stats=stats,
        config=training_config,
        project_name=DEFAULT_WANDB_PROJECT,
        run_name="ppo-training",
        episode_save_dir="recordings/farmtila",
        device=training_config["device"],
        batch_size=training_config["batch_size"],
        buffer_size=training_config["buffer_size"],
        dtype=training_config["dtype"],
        use_wandb=use_wandb,
    )

    simulator = Simulator(
        env=env,
        num_agents=training_config["num_agents"],
        policies=policy_models,
        loss_trainer=loss_trainer,
        agent_factory=agent_factory,
    )

    # Start training
    simulator.train(title="Farmtila Training")


if __name__ == "__main__":
    cli_args = parse_args()
    train(use_wandb=cli_args.wandb == "on")
