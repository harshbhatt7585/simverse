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
from simverse.abstractor.train_utils import (
    build_adam_optimizers,
    build_ppo_training_config,
    compile_policy_models,
    configure_torch_backend,
    resolve_rollout_dtype,
    resolve_torch_device,
)
from simverse.agent.stats import TrainingStats
from simverse.config.policy import PolicySpec
from simverse.envs.maze_race.agent import MazeRaceAgent
from simverse.envs.maze_race.config import MazeRaceConfig
from simverse.envs.maze_race.env import MazeRaceEnv, create_env
from simverse.envs.maze_race.live_server import LiveRenderServer
from simverse.logging_config import training_logger
from simverse.losses.ppo import PPOTrainer
from simverse.policies.simple import SimplePolicy
from simverse.simulator import Simulator
from simverse.wandb_config import DEFAULT_WANDB_PROJECT


def agent_factory(agent_id: int, policy: Policy, env: MazeRaceEnv) -> MazeRaceAgent:
    action_values = np.arange(getattr(env.action_space, "n", 5), dtype=np.int64)
    return MazeRaceAgent(
        agent_id=agent_id,
        action_space=action_values,
        policy=policy,
        name=f"maze_race_agent_{agent_id}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Maze Race PPO agents")
    parser.add_argument("--num-envs", type=int, default=512, help="Parallel environment count")
    parser.add_argument("--episodes", type=int, default=150, help="Training episodes")
    parser.add_argument("--wandb", choices=["on", "off"], default="off")
    parser.add_argument("--compile", choices=["on", "off"], default="on")
    parser.add_argument("--render-server", choices=["on", "off"], default="on")
    parser.add_argument("--render-host", type=str, default="127.0.0.1")
    parser.add_argument("--render-port", type=int, default=8770)
    parser.add_argument("--render-stride", type=int, default=1, help="Stream every Nth frame")
    return parser.parse_args()


def train(
    num_envs: int = 512,
    episodes: int = 150,
    use_wandb: bool = False,
    use_compile: bool = True,
    render_server: bool = True,
    render_host: str = "127.0.0.1",
    render_port: int = 8770,
    render_stride: int = 1,
) -> None:
    device = resolve_torch_device(prefer_mps=True)
    dtype = resolve_rollout_dtype(device, cpu_dtype=torch.bfloat16)
    configure_torch_backend(device)

    config = MazeRaceConfig(
        width=7,
        height=7,
        num_agents=1,
        num_envs=max(1, int(num_envs)),
        max_steps=200,
        win_reward=1.0,
        lose_penalty=1.0,
        draw_reward=0.0,
        policies=[],
    )

    env = create_env(config, num_envs=config.num_envs, device=device, dtype=dtype)

    policy_specs = [
        PolicySpec(
            name=f"maze_race_agent_{agent_id}",
            model=SimplePolicy(obs_space=env.observation_space, action_space=env.action_space),
        )
        for agent_id in range(config.num_agents)
    ]
    env.config.policies = policy_specs

    policy_models = compile_policy_models(
        policy_specs,
        use_compile=use_compile,
        device=device,
    )
    optimizers = build_adam_optimizers(policy_models, lr=3e-4, device=device)

    stats = TrainingStats()
    training_config = build_ppo_training_config(
        num_agents=config.num_agents,
        num_envs=config.num_envs,
        max_steps=config.max_steps,
        episodes=int(episodes),
        training_epochs=1,
        lr=3e-4,
        batch_size=config.num_envs * 2,
        buffer_size=config.num_envs * config.num_agents * 8,
        device=device,
        dtype=dtype,
    )

    live_server = None
    frame_sink = None
    if render_server:
        live_server = LiveRenderServer(
            output_path="recordings/maze_race/live.jsonl",
            host=render_host,
            port=render_port,
            title="Maze Race Live",
            frame_stride=render_stride,
        )
        live_server.start()
        live_server.push_meta(
            {
                "title": "Maze Race Live",
                "env": "maze_race",
                "width": config.width,
                "height": config.height,
                "channels": 5,
            }
        )
        training_logger.info(f"Live render server running at {live_server.url()}")
        frame_sink = live_server.push_frame

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
        run_name="ppo-maze-race",
        episode_save_dir="recordings/maze_race",
        frame_sink=frame_sink,
        device=training_config["device"],
        batch_size=training_config["batch_size"],
        buffer_size=training_config["buffer_size"],
        dtype=training_config["dtype"],
        use_wandb=use_wandb,
    )

    simulator = Simulator(
        env=env,
        num_agents=config.num_agents,
        policies=policy_models,
        loss_trainer=loss_trainer,
        agent_factory=agent_factory,
    )
    try:
        simulator.train(title="Maze Race Training")
    finally:
        if live_server is not None:
            live_server.stop()


if __name__ == "__main__":
    cli_args = parse_args()
    train(
        num_envs=cli_args.num_envs,
        episodes=cli_args.episodes,
        use_wandb=cli_args.wandb == "on",
        use_compile=cli_args.compile == "on",
        render_server=cli_args.render_server == "on",
        render_host=cli_args.render_host,
        render_port=cli_args.render_port,
        render_stride=cli_args.render_stride,
    )
