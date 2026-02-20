from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

if __package__ is None or __package__.startswith("__main__"):
    _src = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_src))

import numpy as np
import torch

from simverse.abstractor.live_render_server import LiveRenderServer
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
from simverse.envs.snake.agent import SnakeAgent
from simverse.envs.snake.config import SnakeConfig
from simverse.envs.snake.env import SnakeEnv, create_env
from simverse.logging_config import training_logger
from simverse.losses.ppo import PPOTrainer
from simverse.policies.simple import SimplePolicy
from simverse.simulator import Simulator
from simverse.wandb_config import DEFAULT_WANDB_PROJECT


def agent_factory(agent_id: int, policy: Policy, env: SnakeEnv) -> SnakeAgent:
    action_values = np.arange(getattr(env.action_space, "n", 4), dtype=np.int64)
    return SnakeAgent(
        agent_id=agent_id,
        action_space=action_values,
        policy=policy,
        name=f"snake_agent_{agent_id}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Snake PPO agent")
    parser.add_argument("--width", type=int, default=15, help="Grid width")
    parser.add_argument("--height", type=int, default=15, help="Grid height")
    parser.add_argument("--num-envs", type=int, default=512, help="Parallel environment count")
    parser.add_argument("--episodes", type=int, default=200, help="Training episodes")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--init-length", type=int, default=3, help="Initial snake length")
    parser.add_argument("--food-reward", type=float, default=5.0)
    parser.add_argument("--crash-penalty", type=float, default=1.0)
    parser.add_argument("--distance-reward-scale", type=float, default=0.01)
    parser.add_argument("--survival-bonus", type=float, default=1.0)
    parser.add_argument("--survival-bonus-every", type=int, default=10)
    parser.add_argument("--training-epochs", type=int, default=3)
    parser.add_argument("--auto-reset-done-envs", choices=["on", "off"], default="on")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wandb", choices=["on", "off"], default="off")
    parser.add_argument("--compile", choices=["on", "off"], default="on")
    parser.add_argument("--render-server", choices=["on", "off"], default="on")
    parser.add_argument("--render-host", type=str, default="127.0.0.1")
    parser.add_argument("--render-port", type=int, default=8770)
    parser.add_argument("--render-stride", type=int, default=1, help="Stream every Nth frame")
    return parser.parse_args()


def train(
    width: int = 15,
    height: int = 15,
    num_envs: int = 512,
    episodes: int = 200,
    max_steps: int = 300,
    init_length: int = 3,
    food_reward: float = 5.0,
    crash_penalty: float = 1.0,
    distance_reward_scale: float = 0.01,
    survival_bonus: float = 1.0,
    survival_bonus_every: int = 10,
    training_epochs: int = 3,
    auto_reset_done_envs: bool = True,
    lr: float = 1e-4,
    seed: int | None = None,
    use_wandb: bool = False,
    use_compile: bool = True,
    render_server: bool = True,
    render_host: str = "127.0.0.1",
    render_port: int = 8770,
    render_stride: int = 1,
) -> None:
    if seed is not None:
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))

    device = resolve_torch_device(prefer_mps=True)
    dtype = resolve_rollout_dtype(device, cpu_dtype=torch.bfloat16)
    configure_torch_backend(device)

    config = SnakeConfig(
        width=max(5, int(width)),
        height=max(5, int(height)),
        num_agents=1,
        num_envs=max(1, int(num_envs)),
        max_steps=max(1, int(max_steps)),
        init_length=max(2, int(init_length)),
        food_reward=float(food_reward),
        crash_penalty=float(crash_penalty),
        distance_reward_scale=float(distance_reward_scale),
        survival_bonus=float(survival_bonus),
        survival_bonus_every=max(1, int(survival_bonus_every)),
        auto_reset_done_envs=bool(auto_reset_done_envs),
        seed=seed,
        policies=[],
    )

    env = create_env(config, num_envs=config.num_envs, device=device, dtype=dtype)

    policy_specs = [
        PolicySpec(
            name="snake_agent_0",
            model=SimplePolicy(
                obs_space=env.observation_space["obs"],
                action_space=env.action_space,
            ),
        )
    ]
    env.config.policies = policy_specs

    policy_models = compile_policy_models(
        policy_specs,
        use_compile=use_compile,
        device=device,
    )
    optimizers = build_adam_optimizers(policy_models, lr=lr, device=device)

    stats = TrainingStats()
    training_config = build_ppo_training_config(
        num_agents=config.num_agents,
        num_envs=config.num_envs,
        max_steps=config.max_steps,
        episodes=int(episodes),
        training_epochs=max(1, int(training_epochs)),
        lr=lr,
        batch_size=config.num_envs * 8,
        buffer_size=config.num_envs * config.num_agents * 16,
        device=device,
        dtype=dtype,
        extras={
            "env": "snake",
            "width": config.width,
            "height": config.height,
        },
    )

    run_name = f"snake-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    live_server = None
    frame_sink = None
    if render_server:
        live_server = LiveRenderServer(
            output_path="recordings/snake/live.jsonl",
            game="snake",
            host=render_host,
            port=render_port,
            title="Snake Live",
            frame_stride=render_stride,
        )
        live_server.start()
        live_server.push_meta(
            {
                "title": "Snake Live",
                "env": "snake",
                "width": config.width,
                "height": config.height,
                "channels": int(env.obs_channels),
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
        run_name=run_name,
        episode_save_dir="recordings/snake",
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
        simulator.train(title="Snake Training")
    finally:
        if live_server is not None:
            live_server.stop()


if __name__ == "__main__":
    cli_args = parse_args()
    train(
        width=cli_args.width,
        height=cli_args.height,
        num_envs=cli_args.num_envs,
        episodes=cli_args.episodes,
        max_steps=cli_args.max_steps,
        init_length=cli_args.init_length,
        food_reward=cli_args.food_reward,
        crash_penalty=cli_args.crash_penalty,
        distance_reward_scale=cli_args.distance_reward_scale,
        survival_bonus=cli_args.survival_bonus,
        survival_bonus_every=cli_args.survival_bonus_every,
        training_epochs=cli_args.training_epochs,
        auto_reset_done_envs=cli_args.auto_reset_done_envs == "on",
        lr=cli_args.lr,
        seed=cli_args.seed,
        use_wandb=cli_args.wandb == "on",
        use_compile=cli_args.compile == "on",
        render_server=cli_args.render_server == "on",
        render_host=cli_args.render_host,
        render_port=cli_args.render_port,
        render_stride=cli_args.render_stride,
    )
