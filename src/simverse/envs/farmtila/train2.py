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
from simverse.agent.stats import TrainingStats
from simverse.config.policy import PolicySpec
from simverse.envs.farmtila.agent import FarmtilaAgent
from simverse.envs.farmtila.config import FarmtilaConfig
from simverse.envs.farmtila.env import FarmtilaEnv, FarmtillaVectorizedEnv
from simverse.envs.farmtila.training_config import build_training_config
from simverse.losses.ppo import PPOTrainer
from simverse.policies.simple import SimplePolicy
from simverse.simulator import Simulator


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
    parser = argparse.ArgumentParser(description="Train Farmtila PPO agents (vectorized env)")
    parser.add_argument(
        "--wandb",
        choices=["on", "off"],
        default="on",
        help="Enable or disable Weights & Biases logging",
    )
    return parser.parse_args()


def train(use_wandb: bool = True):
    training_config = build_training_config(
        num_agents=4,
        num_envs=256,
        max_steps=1000,
        episodes=100,
        training_epochs=1,
        lr=0.001,
        clip_epsilon=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        total_seeds=500,
        batch_size=512,
        buffer_size=50000,
        dtype=torch.float32,
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
    if training_config["num_envs"] > 1:
        env = FarmtillaVectorizedEnv(
            config=config,
            num_envs=training_config["num_envs"],
        )
    else:
        env = FarmtilaEnv(config=config)

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
    optimizers = {
        agent_id: torch.optim.Adam(policy_models[agent_id].parameters(), lr=training_config["lr"])
        for agent_id in range(training_config["num_agents"])
    }

    stats = TrainingStats()

    loss_trainer = PPOTrainer(
        optimizers=optimizers,
        episodes=training_config["episodes"],
        training_epochs=training_config["training_epochs"],
        clip_epsilon=training_config["clip_epsilon"],
        gamma=training_config["gamma"],
        gae_lambda=training_config["gae_lambda"],
        stats=stats,
        config=training_config,
        project_name="simverse-farmtila",
        run_name="ppo-training-vectorized-env",
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

    simulator.train(title="Farmtila Training (VectorizedEnv)")


if __name__ == "__main__":
    cli_args = parse_args()
    train(use_wandb=cli_args.wandb == "on")
