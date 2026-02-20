from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__.startswith("__main__"):
    _src = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_src))

import numpy as np
import torch.nn as nn
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
from simverse.envs.battle_grid.agent import BattleGridAgent
from simverse.envs.battle_grid.config import BattleGridConfig
from simverse.envs.battle_grid.env import BattleGridEnv, create_env
from simverse.losses.ppo import PPOTrainer
from simverse.policies.simple import SimplePolicy
from simverse.simulator import Simulator
from simverse.wandb_config import DEFAULT_WANDB_PROJECT


def agent_factory(agent_id: int, policy: nn.Module, env: BattleGridEnv) -> BattleGridAgent:
    action_values = np.arange(getattr(env.action_space, "n", 6), dtype=np.int64)
    return BattleGridAgent(
        agent_id=agent_id,
        action_space=action_values,
        policy=policy,
        name=f"battle_agent_{agent_id}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Battle Grid PPO agents")
    parser.add_argument("--num-envs", type=int, default=512, help="Parallel environment count")
    parser.add_argument("--episodes", type=int, default=150, help="Training episodes")
    parser.add_argument("--wandb", choices=["on", "off"], default="off")
    parser.add_argument("--compile", choices=["on", "off"], default="on")
    return parser.parse_args()


def train(
    num_envs: int = 512,
    episodes: int = 150,
    use_wandb: bool = False,
    use_compile: bool = True,
) -> None:
    device = resolve_torch_device(prefer_mps=False)
    dtype = resolve_rollout_dtype(device)
    configure_torch_backend(device)

    config = BattleGridConfig(
        width=13,
        height=13,
        num_agents=2,
        num_envs=max(1, int(num_envs)),
        max_steps=200,
        max_health=3,
        attack_damage=1,
        attack_range=1,
        step_penalty=0.01,
        damage_reward=0.05,
        kill_reward=1.0,
        death_penalty=1.0,
        timeout_win_reward=0.5,
        timeout_lose_penalty=0.5,
        draw_reward=0.0,
        policies=[],
    )

    env = create_env(config, num_envs=config.num_envs, device=device, dtype=dtype)

    policy_specs = [
        PolicySpec(
            name=f"battle_agent_{agent_id}",
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
        batch_size=config.num_envs * 4,
        buffer_size=config.num_envs * config.num_agents * 8,
        device=device,
        dtype=dtype,
    )

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
        run_name="ppo-battle-grid",
        episode_save_dir="recordings/battle_grid",
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
    simulator.train(title="Battle Grid Training")


if __name__ == "__main__":
    cli_args = parse_args()
    train(
        num_envs=cli_args.num_envs,
        episodes=cli_args.episodes,
        use_wandb=cli_args.wandb == "on",
        use_compile=cli_args.compile == "on",
    )
