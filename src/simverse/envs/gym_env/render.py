from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

if __package__ is None or __package__.startswith("__main__"):
    _src = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_src))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from simverse.abstractor.policy import Policy
from simverse.envs.gym_env.torch_env import observation_batch_to_chw


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
        return self.action_head(x), self.value_head(x)


def _load_policy_from_checkpoint(
    checkpoint_path: str,
    obs_space: gym.spaces.Box,
    action_space: gym.spaces.Discrete,
    device: str,
) -> torch.nn.Module:
    state = pickle.loads(Path(checkpoint_path).read_bytes())
    agents_state = state.get("agents", [])
    if not agents_state:
        raise ValueError(f"No agent policy found in checkpoint: {checkpoint_path}")

    policy = GymMLPPolicy(obs_space=obs_space, action_space=action_space)
    checkpoint_state_dict = agents_state[0]["policy_state_dict"]
    try:
        policy.load_state_dict(checkpoint_state_dict)
    except RuntimeError:
        normalized_state_dict = {
            (key[len("_orig_mod.") :] if key.startswith("_orig_mod.") else key): value
            for key, value in checkpoint_state_dict.items()
        }
        policy.load_state_dict(normalized_state_dict)
    policy.to(device=device, dtype=torch.float32)
    policy.eval()
    return policy


def _policy_action(policy: torch.nn.Module, obs: np.ndarray, device: str) -> int:
    obs_batch = observation_batch_to_chw(np.expand_dims(np.asarray(obs, dtype=np.float32), axis=0))
    obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits, _ = policy(obs_tensor)
        action = torch.distributions.Categorical(logits=logits.float()).sample().item()
    return int(action)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Gym env episodes with optional recording")
    parser.add_argument("--env-id", type=str, default="CartPole-v1", help="Gymnasium env id")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes to run")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument(
        "--fps", type=int, default=30, help="Frame rate target when using human render"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint path to load policy"
    )
    parser.add_argument("--record", choices=["on", "off"], default="off")
    parser.add_argument("--record-dir", type=str, default="recordings/gym_env/videos")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


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


def render(
    env_id: str = "CartPole-v1",
    episodes: int = 3,
    max_steps: int = 500,
    fps: int = 30,
    checkpoint: str | None = None,
    record: bool = False,
    record_dir: str = "recordings/gym_env/videos",
    seed: int | None = None,
) -> None:
    if episodes <= 0:
        return

    fallback_gif = False
    if record:
        Path(record_dir).mkdir(parents=True, exist_ok=True)
        try:
            base_env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                base_env,
                video_folder=record_dir,
                episode_trigger=lambda _episode_id: True,
                name_prefix=f"{env_id.replace('/', '_')}_render",
            )
        except gym.error.DependencyNotInstalled:
            fallback_gif = True
            env = gym.make(env_id, render_mode="rgb_array")
    else:
        env = gym.make(env_id, render_mode="human")

    if not isinstance(env.action_space, gym.spaces.Discrete):
        env.close()
        raise TypeError(
            "Render script currently supports only discrete action spaces; "
            f"got {type(env.action_space).__name__}"
        )
    if not isinstance(env.observation_space, gym.spaces.Box):
        env.close()
        raise TypeError(
            "Render script currently supports only box observations; "
            f"got {type(env.observation_space).__name__}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = None
    if checkpoint:
        policy = _load_policy_from_checkpoint(
            checkpoint_path=checkpoint,
            obs_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )

    try:
        for episode_idx in range(episodes):
            episode_seed = None if seed is None else int(seed) + episode_idx
            obs, _ = env.reset(seed=episode_seed)
            terminated = False
            truncated = False
            episode_reward = 0.0
            step_count = 0
            frames: list[np.ndarray] = []
            if record and fallback_gif:
                frame = env.render()
                if isinstance(frame, np.ndarray):
                    frames.append(frame)

            while not (terminated or truncated) and step_count < max_steps:
                if policy is None:
                    action = int(env.action_space.sample())
                else:
                    action = _policy_action(policy, obs, device)

                obs, reward, terminated, truncated, _info = env.step(action)
                episode_reward += float(reward)
                step_count += 1
                if record and fallback_gif:
                    frame = env.render()
                    if isinstance(frame, np.ndarray):
                        frames.append(frame)

                if not record and fps > 0:
                    time.sleep(1.0 / fps)

            if record and fallback_gif:
                gif_path = Path(record_dir) / (
                    f"{env_id.replace('/', '_')}_render_ep{episode_idx + 1:03d}.gif"
                )
                _save_gif(frames, gif_path, fps=max(1, fps))

            print(
                f"episode={episode_idx + 1} steps={step_count} "
                f"reward={episode_reward:.3f} done={terminated or truncated}"
            )
    finally:
        env.close()

    if record:
        print(f"Saved recordings to: {record_dir}")


if __name__ == "__main__":
    cli_args = parse_args()
    render(
        env_id=cli_args.env_id,
        episodes=cli_args.episodes,
        max_steps=cli_args.max_steps,
        fps=cli_args.fps,
        checkpoint=cli_args.checkpoint,
        record=cli_args.record == "on",
        record_dir=cli_args.record_dir,
        seed=cli_args.seed,
    )
