"""Prototype for batching policy calls across asynchronous vectorized environments."""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple

import torch

warnings.filterwarnings("ignore", message=".*pynvml.*", category=FutureWarning, module="torch.cuda")

OBS_DIM = 16


ObservationDict = Dict[str, torch.Tensor]


class Policy(Protocol):
    """Callable signature shared by the toy policy classes below."""

    def __call__(self, obs_batch: torch.Tensor) -> torch.Tensor: ...


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

    def _obs(self) -> ObservationDict:
        repeated_time = self.time.view(self.num_envs, 1, 1).expand(-1, self.num_agents, -1)
        return {
            "state": self.state.clone(),
            "prev_action": self.prev_actions.clone(),
            "health": self.health.clone(),
            "time": repeated_time.clone(),
            "env_id": self.env_ids.clone(),
        }

    def reset(self) -> ObservationDict:
        self.state = torch.randn_like(self.state)
        self.prev_actions.zero_()
        self.health.fill_(1.0)
        self.time.zero_()
        return self._obs()

    def step(self, actions: torch.Tensor) -> Tuple[ObservationDict, torch.Tensor, bool]:
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

    def reset(self) -> ObservationDict:
        self._pending_actions = None
        return self.batch.reset()

    def step_async(self, actions: torch.Tensor) -> None:
        if self._pending_actions is not None:
            raise RuntimeError("step_async called before previous step_wait")
        self._pending_actions = actions

    def step_wait(self) -> Tuple[ObservationDict, torch.Tensor, bool]:
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


def build_policy(agent_id: int) -> Policy:
    if agent_id % 3 == 0:
        return LinearPolicy(OBS_DIM, seed=agent_id)
    if agent_id % 3 == 1:
        hidden_dim = 32 + agent_id
        return TwoLayerPolicy(OBS_DIM, hidden_dim=hidden_dim, seed=agent_id)
    return SinusoidPolicy(OBS_DIM, seed=agent_id)


def policy_to_device(policy: Policy, device: torch.device) -> Policy:
    for name, value in vars(policy).items():
        if isinstance(value, torch.Tensor):
            setattr(policy, name, value.to(device))
    return policy


def _format_number(num: float, precision: int = 2) -> str:
    """Format number with appropriate units."""
    if num >= 1e6:
        return f"{num / 1e6:.{precision}f}M"
    if num >= 1e3:
        return f"{num / 1e3:.{precision}f}K"
    return f"{num:.{precision}f}"


def _progress_bar(current: int, total: int, width: int = 30) -> str:
    """Generate a progress bar string."""
    filled = int(width * current / max(total, 1))
    bar = "█" * filled + "░" * (width - filled)
    percent = 100 * current / max(total, 1)
    return f"{bar} {percent:.1f}%"


def run_async_vectorized_demo(
    num_envs: int,
    num_agents: int,
    rollout_steps: int,
    device: torch.device,
) -> None:
    # Header
    print("\n" + "=" * 70)
    print("🚀 Async Vectorized Environment Demo")
    print("=" * 70)
    print(f"  Environments: {num_envs}")
    print(f"  Agents per env: {num_agents}")
    print(f"  Total agents: {num_envs * num_agents}")
    print(f"  Rollout steps: {rollout_steps}")
    print(f"  Device: {device}")
    print(f"  Observation dim: {OBS_DIM}")
    print("=" * 70)

    # Initialize
    start_time = time.perf_counter()
    env = AsyncVectorizedEnv(num_envs, num_agents, device)
    policies: List[Policy] = [
        policy_to_device(build_policy(agent_id), device) for agent_id in range(num_agents)
    ]

    print("\n📋 Policy Configuration:")
    policy_counts: Dict[str, int] = {}
    for agent_id, policy in enumerate(policies):
        policy_name = policy.__class__.__name__
        policy_counts[policy_name] = policy_counts.get(policy_name, 0) + 1
        print(f"  Agent {agent_id:2d}: {policy_name}")
    print("\n📊 Policy Distribution:")
    for policy_name, count in sorted(policy_counts.items()):
        print(f"  {policy_name:20s}: {count:2d} agents")

    obs = env.reset()
    last_ts = time.perf_counter()
    total_steps = 0
    total_time = 0.0
    avg_throughput = 0.0
    step_times: List[float] = []
    step_throughputs: List[float] = []

    print("\n" + "─" * 70)
    print("🔄 Running Rollout...")
    print("─" * 70)

    for step in range(rollout_steps):
        step_start = time.perf_counter()
        state_obs = obs["state"]  # [num_envs, num_agents, obs_dim]

        # Policy inference
        inference_start = time.perf_counter()
        per_agent_actions: List[torch.Tensor] = []
        for agent_id, policy in enumerate(policies):
            obs_for_agent = state_obs[:, agent_id, :]
            actions = policy(obs_for_agent)
            per_agent_actions.append(actions)
        inference_time = time.perf_counter() - inference_start

        batched_actions = torch.stack(per_agent_actions, dim=1)

        # Environment step
        env_start = time.perf_counter()
        env.step_async(batched_actions)
        obs, rewards, done = env.step_wait()
        env_time = time.perf_counter() - env_start

        # Timing
        now = time.perf_counter()
        step_elapsed = now - step_start
        total_elapsed = now - last_ts
        total_steps += num_envs * num_agents
        total_time += total_elapsed
        step_times.append(step_elapsed)

        agent_steps_per_sec = (num_envs * num_agents) / max(total_elapsed, 1e-6)
        step_throughputs.append(agent_steps_per_sec)
        avg_throughput = total_steps / max(total_time, 1e-6)

        # Progress bar
        progress = _progress_bar(step + 1, rollout_steps)

        # Log step details
        reward_mean = rewards.mean().item() if isinstance(rewards, torch.Tensor) else float(rewards)
        reward_std = rewards.std().item() if isinstance(rewards, torch.Tensor) else 0.0

        print(
            f"Step {step + 1:3d}/{rollout_steps} │ {progress} │ "
            f"Throughput: {agent_steps_per_sec:7.1f} agent-steps/s │ "
            f"Avg: {avg_throughput:7.1f} agent-steps/s"
        )
        print(
            f"  ├─ Inference: {inference_time * 1000:6.2f}ms │ "
            f"Env step: {env_time * 1000:6.2f}ms │ "
            f"Total: {step_elapsed * 1000:6.2f}ms"
        )
        print(
            f"  ├─ Reward: mean={reward_mean:7.3f}, std={reward_std:7.3f} │ "
            f"Actions shape: {list(batched_actions.shape)}"
        )

        last_ts = now

    # Summary
    total_wall_time = time.perf_counter() - start_time
    print("\n" + "─" * 70)
    print("📈 Summary Statistics")
    print("─" * 70)

    if not step_times:
        print("  No rollout steps executed.")
        print("=" * 70 + "\n")
        return

    avg_step_ms = sum(step_times) / len(step_times) * 1000
    print(f"  Total agent-steps: {_format_number(float(total_steps))}")
    print(f"  Total wall time: {total_wall_time:.3f}s")
    print(f"  Average throughput: {_format_number(avg_throughput)} agent-steps/s")
    print(f"  Peak throughput: {_format_number(max(step_throughputs))} agent-steps/s")
    print(f"  Min throughput: {_format_number(min(step_throughputs))} agent-steps/s")
    print(f"  Average step time: {avg_step_ms:.2f}ms")
    print(f"  Min step time: {min(step_times) * 1000:.2f}ms")
    print(f"  Max step time: {max(step_times) * 1000:.2f}ms")
    print("=" * 70 + "\n")


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
        print("⚠️  MPS requested but not available; falling back to CPU")
        requested_device = "cpu"
    device = torch.device(requested_device)
    run_async_vectorized_demo(
        num_envs=args.num_envs,
        num_agents=args.num_agents,
        rollout_steps=args.rollout_steps,
        device=device,
    )
