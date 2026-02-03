"""Prototype for batching policy calls across asynchronous vectorized environments."""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

warnings.filterwarnings("ignore", message=".*pynvml.*", category=FutureWarning, module="torch.cuda")

OBS_DIM = 16


@dataclass
class VectorizedEnvBatch:
    """Fully vectorized multi-agent environment that lives entirely on a device."""

    num_envs: int
    num_agents: int
    device: torch.device

    def __post_init__(self) -> None:
        self.state = torch.randn(self.num_envs, self.num_agents, OBS_DIM, device=self.device)
        self.prev_actions = torch.zeros_like(self.state)
        self.health = torch.ones(self.num_envs, self.num_agents, 1, device=self.device)
        env_ids = torch.arange(self.num_envs, device=self.device).float()
        self.env_ids = env_ids.view(self.num_envs, 1, 1).expand(-1, self.num_agents, 1)
        self.time = torch.zeros(self.num_envs, 1, device=self.device)

    def _obs(self) -> Dict[str, torch.Tensor]:
        repeated_time = self.time.view(self.num_envs, 1, 1).expand(-1, self.num_agents, -1)
        return {
            "state": self.state.clone(),
            "prev_action": self.prev_actions.clone(),
            "health": self.health.clone(),
            "time": repeated_time.clone(),
            "env_id": self.env_ids.clone(),
        }

    def reset(self) -> Dict[str, torch.Tensor]:
        self.state = torch.randn_like(self.state)
        self.prev_actions.zero_()
        self.health.fill_(1.0)
        self.time.zero_()
        return self._obs()

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, bool]:
        noise = 0.05 * torch.randn_like(self.state)
        self.state = self.state + actions + noise
        self.prev_actions = actions.clone()
        self.health = torch.clamp(self.health - 0.01 * actions.norm(dim=2, keepdim=True), min=0.0)
        self.time += 0.05
        reward = -self.state.norm(dim=2)
        done = False
        return self._obs(), reward, done


class AsyncVectorizedEnv:
    """Mimics Gym's AsyncVectorEnv API but keeps everything vectorized on a device."""

    def __init__(self, num_envs: int, num_agents: int, device: torch.device) -> None:
        self.batch = VectorizedEnvBatch(num_envs, num_agents, device)
        self._pending_actions: Optional[torch.Tensor] = None

    def reset(self) -> Dict[str, torch.Tensor]:
        self._pending_actions = None
        return self.batch.reset()

    def step_async(self, actions: torch.Tensor) -> None:
        if self._pending_actions is not None:
            raise RuntimeError("step_async called before previous step_wait")
        self._pending_actions = actions

    def step_wait(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, bool]:
        if self._pending_actions is None:
            raise RuntimeError("step_wait called without pending actions")
        obs, reward, done = self.batch.step(self._pending_actions)
        self._pending_actions = None
        return obs, reward, done


class LinearPolicy:
    def __init__(self, obs_dim: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(obs_dim, obs_dim, generator=generator)
        self.bias = torch.randn(obs_dim, generator=generator)

    def __call__(self, obs_batch: torch.Tensor) -> torch.Tensor:
        logits = obs_batch @ self.weight + self.bias
        return torch.tanh(logits)


class TwoLayerPolicy:
    def __init__(self, obs_dim: int, hidden_dim: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.w1 = torch.randn(obs_dim, hidden_dim, generator=generator) / hidden_dim**0.5
        self.b1 = torch.zeros(hidden_dim)
        self.w2 = torch.randn(hidden_dim, obs_dim, generator=generator) / obs_dim**0.5
        self.b2 = torch.zeros(obs_dim)

    def __call__(self, obs_batch: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(obs_batch @ self.w1 + self.b1)
        logits = hidden @ self.w2 + self.b2
        return torch.tanh(logits)


class SinusoidPolicy:
    def __init__(self, obs_dim: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.phase = torch.randn(obs_dim, generator=generator)
        self.scale = torch.randn(obs_dim, generator=generator)

    def __call__(self, obs_batch: torch.Tensor) -> torch.Tensor:
        return torch.sin(obs_batch + self.phase) * torch.tanh(self.scale)


def build_policy(agent_id: int) -> object:
    if agent_id % 3 == 0:
        return LinearPolicy(OBS_DIM, seed=agent_id)
    if agent_id % 3 == 1:
        hidden_dim = 32 + agent_id
        return TwoLayerPolicy(OBS_DIM, hidden_dim=hidden_dim, seed=agent_id)
    return SinusoidPolicy(OBS_DIM, seed=agent_id)


def policy_to_device(policy: object, device: torch.device) -> object:
    for name, value in vars(policy).items():
        if isinstance(value, torch.Tensor):
            setattr(policy, name, value.to(device))
    return policy


def run_async_vectorized_demo(
    num_envs: int,
    num_agents: int,
    rollout_steps: int,
    device: torch.device,
) -> None:
    env = AsyncVectorizedEnv(num_envs, num_agents, device)
    policies = [policy_to_device(build_policy(agent_id), device) for agent_id in range(num_agents)]
    print("Assigned policies:")
    for agent_id, policy in enumerate(policies):
        print(f"  Agent {agent_id}: {policy.__class__.__name__}")

    obs = env.reset()
    last_ts = time.perf_counter()
    total_steps = 0
    total_time = 0.0
    for step in range(rollout_steps):
        state_obs = obs["state"]  # [num_envs, num_agents, obs_dim]
        agents_obs = state_obs.permute(1, 2, 0)
        print(f"Step {step}: agents_obs shape {agents_obs.shape}")

        per_agent_actions: List[torch.Tensor] = []
        for agent_id, policy in enumerate(policies):
            obs_for_agent = state_obs[:, agent_id, :]
            actions = policy(obs_for_agent)
            per_agent_actions.append(actions)

        batched_actions = torch.stack(per_agent_actions, dim=1)
        env.step_async(batched_actions)
        obs, _, _ = env.step_wait()

        now = time.perf_counter()
        elapsed = max(now - last_ts, 1e-6)
        total_steps += num_envs * num_agents
        total_time += elapsed
        agent_steps_per_sec = (num_envs * num_agents) / elapsed
        avg_steps = total_steps / max(total_time, 1e-6)
        print(
            f"Step {step}: actions shape {batched_actions.shape} | "
            f"{agent_steps_per_sec:.1f} agent-steps/sec (avg {avg_steps:.1f})"
        )
        last_ts = now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async vectorized env demo")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--rollout-steps", type=int, default=5)
    parser.add_argument(
        "--device",
        choices=["cpu", "mps"],
        default="cpu",
        help="Device to run batched policies on",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    requested_device = args.device
    if requested_device == "mps" and not torch.backends.mps.is_available():
        print("MPS requested but not available; falling back to CPU")
        requested_device = "cpu"
    device = torch.device(requested_device)
    run_async_vectorized_demo(
        num_envs=args.num_envs,
        num_agents=args.num_agents,
        rollout_steps=args.rollout_steps,
        device=device,
    )
