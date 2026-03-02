from __future__ import annotations

from datetime import datetime

import numpy as np
import torch
from simverse.envs.snake.agent import SnakeAgent
from simverse.envs.snake.config import SnakeConfig
from simverse.envs.snake.env import SnakeEnv, create_env
from simverse.logging_config import training_logger
from simverse.policies.simple import SimplePolicy
from simverse.render.live_server import LiveRenderServer
from simverse.training.utils import (
    build_ppo_training_config,
    configure_torch_backend,
    resolve_rollout_dtype,
    resolve_torch_device,
    run_ppo_training,
)


def agent_factory(agent_id: int, policy: torch.nn.Module, env: SnakeEnv) -> SnakeAgent:
    action_values = np.arange(getattr(env.action_space, "n", 4), dtype=np.int64)
    return SnakeAgent(
        agent_id=agent_id,
        action_space=action_values,
        policy=policy,
        name=f"snake_agent_{agent_id}",
    )


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

    try:
        run_ppo_training(
            env=env,
            training_config=training_config,
            agent_factory=agent_factory,
            policy_factory=lambda _obs_space, action_space: SimplePolicy(
                obs_space=env.observation_space["obs"],
                action_space=action_space,
            ),
            title="Snake Training",
            run_name=run_name,
            episode_save_dir="recordings/snake",
            use_wandb=use_wandb,
            use_compile=use_compile,
            policy_name_prefix="snake_agent",
            frame_sink=frame_sink,
        )
    finally:
        if live_server is not None:
            live_server.stop()


if __name__ == "__main__":
    train()
