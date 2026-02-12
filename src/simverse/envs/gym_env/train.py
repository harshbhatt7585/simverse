from __future__ import annotations

import argparse
import inspect
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

if __package__ is None or __package__.startswith("__main__"):
    _src = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_src))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from simverse.abstractor.agent import SimAgent
from simverse.abstractor.policy import Policy
from simverse.agent.stats import TrainingStats
from simverse.config.policy import PolicySpec
from simverse.envs.gym_env.torch_env import GymTorchConfig, GymTorchEnv, observation_batch_to_chw
from simverse.losses.ppo import PPOTrainer
from simverse.simulator import Simulator


class GymMLPPolicy(Policy):
    def __init__(self, obs_space: gym.spaces.Box, action_space: gym.spaces.Discrete) -> None:
        super().__init__()
        input_dim = int(np.prod(obs_space.shape))
        hidden_dim = 256
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.action_head = nn.Linear(hidden_dim, action_space.n)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        target = self.action_head.weight
        if obs.device != target.device or obs.dtype != target.dtype:
            obs = obs.to(device=target.device, dtype=target.dtype)
        x = self.encoder(obs)
        logits = self.action_head(x)
        value = self.value_head(x)
        return logits, value


class GymAgent(SimAgent):
    def __init__(
        self,
        agent_id: int,
        action_count: int,
        policy: Optional[Policy] = None,
        name: str | None = None,
    ) -> None:
        action_space = np.arange(action_count, dtype=np.int64)
        super().__init__(
            name=name or f"gym_agent_{agent_id}", action_space=action_space, policy=policy
        )
        self.agent_id = agent_id
        self.reward = 0.0
        self.memory: dict = {}

    def action(self, obs: np.ndarray) -> np.ndarray:
        if self.policy is not None:
            with torch.no_grad():
                obs_batch = observation_batch_to_chw(np.expand_dims(obs, axis=0))
                obs_tensor = torch.from_numpy(obs_batch).float()
                logits, _ = self.policy(obs_tensor)
                action = torch.distributions.Categorical(logits=logits.float()).sample()
            return action.detach().cpu().numpy()
        return np.array([np.random.choice(self.action_space)], dtype=np.int64)

    def info(self) -> dict:
        return {"agent_id": self.agent_id, "reward": self.reward}

    def reset(self) -> None:
        self.reward = 0.0
        self.memory.clear()

    def get_action_space(self) -> np.ndarray:
        return self.action_space

    def get_memory(self) -> dict:
        return self.memory

    def current_state(self) -> np.ndarray:
        return np.array([self.reward], dtype=np.float32)

    def get_policy(self):
        return self.policy

    def set_policy(self, policy) -> None:
        self.policy = policy


def agent_factory(agent_id: int, policy: Policy, env: GymTorchEnv) -> GymAgent:
    return GymAgent(
        agent_id=agent_id,
        action_count=env.action_space.n,
        policy=policy,
        name=f"{env.config.env_id}_agent_{agent_id}",
    )


def _record_policy_video(
    *,
    env_id: str,
    policy: torch.nn.Module,
    device: str,
    max_steps: int,
    episodes: int,
    output_dir: str,
    seed: int | None,
) -> None:
    if episodes <= 0:
        return

    video_dir = Path(output_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    policy.eval()
    policy.to(device=device, dtype=torch.float32)

    def _rollout_episode(env: gym.Env, episode_seed: int | None) -> list[np.ndarray]:
        obs, _ = env.reset(seed=episode_seed)
        terminated = False
        truncated = False
        steps = 0
        frames: list[np.ndarray] = []
        frame = env.render()
        if isinstance(frame, np.ndarray):
            frames.append(frame)
        while not (terminated or truncated) and steps < max_steps:
            obs_batch = observation_batch_to_chw(
                np.expand_dims(np.asarray(obs, dtype=np.float32), axis=0)
            )
            obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, _ = policy(obs_tensor)
                action = torch.distributions.Categorical(logits=logits.float()).sample().item()
            obs, _reward, terminated, truncated, _info = env.step(action)
            steps += 1
            frame = env.render()
            if isinstance(frame, np.ndarray):
                frames.append(frame)
        return frames

    def _save_gif(frames: list[np.ndarray], output_path: Path, fps: int = 30) -> None:
        if not frames:
            return
        from PIL import Image

        pil_frames = [Image.fromarray(np.asarray(frame, dtype=np.uint8)) for frame in frames]
        duration_ms = max(1, int(round(1000.0 / max(fps, 1))))
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
        )

    try:
        base_env = gym.make(env_id, render_mode="rgb_array")
        video_env = gym.wrappers.RecordVideo(
            base_env,
            video_folder=str(video_dir),
            episode_trigger=lambda _episode_id: True,
            name_prefix=f"{env_id.replace('/', '_')}_ppo",
        )
    except gym.error.DependencyNotInstalled:
        fallback_env = gym.make(env_id, render_mode="rgb_array")
        try:
            for episode_idx in range(episodes):
                episode_seed = None if seed is None else int(seed) + episode_idx
                frames = _rollout_episode(fallback_env, episode_seed)
                gif_path = video_dir / f"{env_id.replace('/', '_')}_ppo_ep{episode_idx + 1:03d}.gif"
                _save_gif(frames, gif_path)
        finally:
            fallback_env.close()
        return

    try:
        for episode_idx in range(episodes):
            episode_seed = None if seed is None else int(seed) + episode_idx
            _rollout_episode(video_env, episode_seed)
    finally:
        video_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Gymnasium env with Simverse PPO")
    parser.add_argument("--env-id", type=str, default="CartPole-v1", help="Gymnasium env id")
    parser.add_argument("--num-envs", type=int, default=512, help="Parallel environment count")
    parser.add_argument("--episodes", type=int, default=120, help="Training episodes")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--wandb", choices=["on", "off"], default="off")
    parser.add_argument("--compile", choices=["on", "off"], default="on")
    parser.add_argument("--record-video", choices=["on", "off"], default="off")
    parser.add_argument("--video-episodes", type=int, default=1)
    parser.add_argument("--video-dir", type=str, default="recordings/gym_env/videos")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def train(
    env_id: str = "CartPole-v1",
    num_envs: int = 512,
    episodes: int = 120,
    max_steps: int = 500,
    lr: float = 3e-4,
    use_wandb: bool = False,
    use_compile: bool = True,
    seed: int | None = None,
    record_video: bool = False,
    video_episodes: int = 1,
    video_dir: str = "recordings/gym_env/videos",
) -> None:
    if seed is not None:
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))

    device = "cuda" if torch.cuda.is_available() else "mps"
    dtype = torch.float16 if device == "cuda" else torch.bfloat16

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    config = GymTorchConfig(
        env_id=env_id,
        num_agents=1,
        num_envs=max(1, int(num_envs)),
        max_steps=max(1, int(max_steps)),
        seed=seed,
        policies=[],
    )

    env = GymTorchEnv(config=config, num_envs=config.num_envs, device=device, dtype=dtype)

    policy_specs = [
        PolicySpec(
            name=f"{env_id}_agent_0",
            model=GymMLPPolicy(obs_space=env.observation_space, action_space=env.action_space),
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
        "env_id": env_id,
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

    resolved_project_name = "simverse-gym"
    resolved_run_name = (
        f"{env_id.replace('/', '_').lower()}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        project_name=resolved_project_name,
        run_name=resolved_run_name,
        episode_save_dir="recordings/gym_env",
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
        simulator.train(title=f"{env_id} Training")
    finally:
        env.close()

    if record_video:
        _record_policy_video(
            env_id=env_id,
            policy=policy_models[0],
            device=device,
            max_steps=config.max_steps,
            episodes=max(1, int(video_episodes)),
            output_dir=video_dir,
            seed=seed,
        )


if __name__ == "__main__":
    cli_args = parse_args()
    train(
        env_id=cli_args.env_id,
        num_envs=cli_args.num_envs,
        episodes=cli_args.episodes,
        max_steps=cli_args.max_steps,
        lr=cli_args.lr,
        use_wandb=cli_args.wandb == "on",
        use_compile=cli_args.compile == "on",
        seed=cli_args.seed,
        record_video=cli_args.record_video == "on",
        video_episodes=cli_args.video_episodes,
        video_dir=cli_args.video_dir,
    )
