from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__.startswith("__main__"):
    _src = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_src))

import numpy as np
import torch

from simverse.abstractor.policy import Policy
from simverse.agent.stats import TrainingStats
from simverse.config.policy import PolicySpec
from simverse.envs.tennis.agent import TennisAgent
from simverse.envs.tennis.config import TennisConfig
from simverse.envs.tennis.torch_env import TennisTorchEnv
from simverse.envs.tennis.training_config import build_training_config
from simverse.losses.ppo import PPOTrainer
from simverse.policies.centralized_critic import CentralizedCritic
from simverse.policies.simple import SimplePolicy
from simverse.simulator import Simulator


def agent_factory(agent_id: int, policy: Policy, env: TennisTorchEnv) -> TennisAgent:
    action_values = np.arange(getattr(env.action_space, "n", 18), dtype=np.int64)
    return TennisAgent(
        agent_id=agent_id,
        action_space=action_values,
        policy=policy,
        name=f"tennis_agent_{agent_id}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Tennis PPO agents (torch env)")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=64,
        help="Number of parallel tennis environment copies to run",
    )
    parser.add_argument(
        "--wandb",
        choices=["on", "off"],
        default="on",
        help="Enable or disable Weights & Biases logging",
    )
    return parser.parse_args()


def train(use_wandb: bool = True, num_envs: int = 64) -> None:
    training_config = build_training_config(
        num_agents=2,
        num_envs=num_envs,
        episodes=100,
        training_epochs=1,
        lr=0.0003,
        clip_epsilon=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.01,
        normalize_advantages=True,
        frame_skip=4,
        obs_resize=84,
        use_grayscale=False,
        batch_size=None,
        buffer_size=None,
        dtype=torch.bfloat16,
    )
    training_config["ctde"] = True
    training_config["torch_fastpath"] = True

    config = TennisConfig(
        num_agents=training_config["num_agents"],
        num_envs=training_config["num_envs"],
        max_steps=training_config["max_steps"],
        frame_skip=training_config["frame_skip"],
        obs_resize=training_config["obs_resize"],
        use_grayscale=training_config["use_grayscale"],
        policies=[],
    )
    env = TennisTorchEnv(
        config=config,
        num_envs=training_config["num_envs"],
        device=training_config["device"],
        dtype=training_config["dtype"],
    )

    actor_obs_space = (
        env.local_observation_space if training_config["ctde"] else env.observation_space
    )

    policy_specs = [
        PolicySpec(
            name=f"simple_agent_{agent_id}",
            model=SimplePolicy(
                obs_space=actor_obs_space,
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
    centralized_critic = CentralizedCritic(obs_space=env.observation_space)
    centralized_critic_optimizer = torch.optim.Adam(
        centralized_critic.parameters(),
        lr=training_config["lr"],
    )

    stats = TrainingStats()

    loss_trainer = PPOTrainer(
        optimizers=optimizers,
        centralized_critic=centralized_critic if training_config["ctde"] else None,
        centralized_critic_optimizer=(
            centralized_critic_optimizer if training_config["ctde"] else None
        ),
        episodes=training_config["episodes"],
        training_epochs=training_config["training_epochs"],
        clip_epsilon=training_config["clip_epsilon"],
        gamma=training_config["gamma"],
        gae_lambda=training_config["gae_lambda"],
        stats=stats,
        config=training_config,
        project_name="simverse-tennis",
        run_name="ppo-training-torch-env",
        episode_save_dir="recordings/tennis",
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

    simulator.train(title="Tennis Training (TorchEnv)")


if __name__ == "__main__":
    cli_args = parse_args()
    train(use_wandb=cli_args.wandb == "on", num_envs=cli_args.num_envs)
