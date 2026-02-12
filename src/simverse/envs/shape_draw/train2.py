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
from simverse.envs.shape_draw.agent import ShapeDrawAgent
from simverse.envs.shape_draw.config import ShapeDrawConfig
from simverse.envs.shape_draw.env import ShapeDrawEnv, ShapeDrawVectorizedEnv
from simverse.envs.shape_draw.training_config import build_training_config
from simverse.losses.ppo import PPOTrainer
from simverse.policies.simple import SimplePolicy
from simverse.simulator import Simulator


def agent_factory(
    agent_id: int,
    policy: Policy,
    env: ShapeDrawEnv | ShapeDrawVectorizedEnv,
) -> ShapeDrawAgent:
    action_values = np.arange(getattr(env.action_space, "n", 7), dtype=np.int64)
    return ShapeDrawAgent(
        agent_id=agent_id,
        action_space=action_values,
        policy=policy,
        name=f"shape_draw_agent_{agent_id}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ShapeDraw PPO agent (vectorized env)")
    parser.add_argument("--num-envs", type=int, default=64, help="Parallel environment count")
    parser.add_argument("--wandb", choices=["on", "off"], default="off")
    parser.add_argument(
        "--recording",
        choices=["on", "off"],
        default="on",
        help="Enable or disable episode recording JSON dumps",
    )
    parser.add_argument(
        "--recording-dir",
        type=str,
        default="recordings/shape_draw",
        help="Directory to write episode recordings when --recording on",
    )
    return parser.parse_args()


def train(
    use_wandb: bool = False,
    num_envs: int = 64,
    recording: bool = True,
    recording_dir: str = "recordings/shape_draw",
) -> None:
    training_config = build_training_config(num_agents=1, num_envs=num_envs)

    config = ShapeDrawConfig(
        width=64,
        height=64,
        num_agents=training_config["num_agents"],
        num_envs=training_config["num_envs"],
        max_steps=training_config["max_steps"],
        policies=[],
    )

    if training_config["num_envs"] > 1:
        env = ShapeDrawVectorizedEnv(config=config, num_envs=training_config["num_envs"])
    else:
        env = ShapeDrawEnv(config=config)

    policy_specs = [
        PolicySpec(
            name="shape_draw_agent_0",
            model=SimplePolicy(obs_space=env.observation_space, action_space=env.action_space),
        )
    ]
    env.config.policies = policy_specs

    policy_models = [ps.model for ps in env.config.policies]
    optimizers = {0: torch.optim.Adam(policy_models[0].parameters(), lr=training_config["lr"])}

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
        project_name="simverse-shape-draw",
        run_name="ppo-shape-draw-vectorized",
        episode_save_dir=recording_dir if recording else None,
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

    simulator.train(title="Shape Draw Training (VectorizedEnv)")


if __name__ == "__main__":
    cli_args = parse_args()
    train(
        use_wandb=cli_args.wandb == "on",
        num_envs=cli_args.num_envs,
        recording=cli_args.recording == "on",
        recording_dir=cli_args.recording_dir,
    )
