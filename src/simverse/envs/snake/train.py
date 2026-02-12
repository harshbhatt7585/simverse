from __future__ import annotations

import argparse
import inspect
import sys
from datetime import datetime
from pathlib import Path

if __package__ is None or __package__.startswith("__main__"):
    _src = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_src))

import numpy as np
import torch

from simverse.abstractor.policy import Policy
from simverse.agent.stats import TrainingStats
from simverse.config.policy import PolicySpec
from simverse.envs.snake.agent import SnakeAgent
from simverse.envs.snake.config import SnakeConfig
from simverse.envs.snake.torch_env import SnakeTorchEnv
from simverse.losses.ppo import PPOTrainer
from simverse.policies.simple import SimplePolicy
from simverse.simulator import Simulator


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def agent_factory(agent_id: int, policy: Policy, env: SnakeTorchEnv) -> SnakeAgent:
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
    parser.add_argument("--food-reward", type=float, default=1.0)
    parser.add_argument("--crash-penalty", type=float, default=2.0)
    parser.add_argument("--distance-reward-scale", type=float, default=0.05)
    parser.add_argument("--survival-bonus", type=float, default=1.0)
    parser.add_argument("--survival-bonus-every", type=int, default=10)
    parser.add_argument("--auto-reset-done-envs", choices=["on", "off"], default="on")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wandb", choices=["on", "off"], default="off")
    parser.add_argument("--compile", choices=["on", "off"], default="on")
    return parser.parse_args()


def train(
    width: int = 15,
    height: int = 15,
    num_envs: int = 512,
    episodes: int = 200,
    max_steps: int = 300,
    init_length: int = 3,
    food_reward: float = 1.0,
    crash_penalty: float = 1.0,
    distance_reward_scale: float = 0.05,
    survival_bonus: float = 1.0,
    survival_bonus_every: int = 10,
    auto_reset_done_envs: bool = True,
    lr: float = 3e-4,
    seed: int | None = None,
    use_wandb: bool = False,
    use_compile: bool = True,
) -> None:
    if seed is not None:
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))

    device = _resolve_device()
    dtype = torch.float16 if device == "cuda" else torch.bfloat16

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

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

    env = SnakeTorchEnv(
        config=config,
        num_envs=config.num_envs,
        device=device,
        dtype=dtype,
    )

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

    policy_models = [ps.model for ps in policy_specs]
    if use_compile and hasattr(torch, "compile") and device == "cuda":
        policy_models = [torch.compile(model, mode="max-autotune") for model in policy_models]

    adam_kwargs: dict[str, object] = {}
    if "fused" in inspect.signature(torch.optim.Adam).parameters and device == "cuda":
        adam_kwargs["fused"] = True
    optimizers = {
        agent_id: torch.optim.Adam(policy_models[agent_id].parameters(), lr=lr, **adam_kwargs)
        for agent_id in range(config.num_agents)
    }

    stats = TrainingStats()
    training_config = {
        "env": "snake",
        "width": config.width,
        "height": config.height,
        "num_agents": config.num_agents,
        "num_envs": config.num_envs,
        "max_steps": config.max_steps,
        "episodes": int(episodes),
        "training_epochs": 1,
        "clip_epsilon": 0.2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "lr": lr,
        "batch_size": config.num_envs * 8,
        "buffer_size": config.num_envs * config.num_agents * 16,
        "device": device,
        "dtype": dtype,
        "torch_fastpath": True,
    }

    run_name = f"snake-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    loss_trainer = PPOTrainer(
        optimizers=optimizers,
        episodes=training_config["episodes"],
        training_epochs=training_config["training_epochs"],
        clip_epsilon=training_config["clip_epsilon"],
        gamma=training_config["gamma"],
        gae_lambda=training_config["gae_lambda"],
        stats=stats,
        config=training_config,
        project_name="simverse-snake",
        run_name=run_name,
        episode_save_dir="recordings/snake",
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

    simulator.train(title="Snake Training")


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
        auto_reset_done_envs=cli_args.auto_reset_done_envs == "on",
        lr=cli_args.lr,
        seed=cli_args.seed,
        use_wandb=cli_args.wandb == "on",
        use_compile=cli_args.compile == "on",
    )
