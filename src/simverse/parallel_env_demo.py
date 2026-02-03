"""Prototype for batching policy calls across parallel environment workers."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

warnings.filterwarnings("ignore", message=".*pynvml.*", category=FutureWarning, module="torch.cuda")


OBS_DIM = 16


@dataclass
class MultiAgentDummyEnv:
    """Environment replica that contains the same agents as every other env."""

    env_id: int
    num_agents: int

    def __post_init__(self) -> None:
        self.state = torch.randn(self.num_agents, OBS_DIM)
        self.prev_actions = torch.zeros(self.num_agents, OBS_DIM)
        self.health = torch.ones(self.num_agents)

    def _build_obs(self) -> Dict[str, torch.Tensor]:
        return {
            "state": self.state.clone(),
            "prev_action": self.prev_actions.clone(),
            "health": self.health.clone(),
            "time": torch.full((self.num_agents, 1), float(time.time() % 1000)),
            "env_id": torch.full((self.num_agents, 1), float(self.env_id)),
        }

    def reset(self) -> Dict[str, torch.Tensor]:
        self.state = torch.randn(self.num_agents, OBS_DIM)
        self.prev_actions.zero_()
        self.health = torch.ones(self.num_agents)
        return self._build_obs()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, bool, Dict[str, float]]:
        noise = 0.05 * torch.randn_like(self.state)
        self.state = self.state + actions + noise
        self.prev_actions = actions.clone()
        self.health = torch.clamp(self.health - 0.01 * actions.norm(dim=1), min=0.0)
        reward = -self.state.norm(dim=1)
        done = False
        info = {"env_id": self.env_id, "reward": reward.mean().item()}
        return self._build_obs(), reward, done, info


class LinearPolicy:
    """Single-layer policy that outputs continuous actions."""

    def __init__(self, obs_dim: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(obs_dim, obs_dim, generator=generator)
        self.bias = torch.randn(obs_dim, generator=generator)

    def __call__(self, obs_batch: torch.Tensor) -> torch.Tensor:
        logits = obs_batch @ self.weight + self.bias
        return torch.tanh(logits)


class TwoLayerPolicy:
    """Simple MLP policy with one hidden layer."""

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
    """Policy that applies sinusoidal transformation to observations."""

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
        hidden_dim = 8 + agent_id
        return TwoLayerPolicy(OBS_DIM, hidden_dim=hidden_dim, seed=agent_id)
    return SinusoidPolicy(OBS_DIM, seed=agent_id)


def policy_to_device(policy: object, device: torch.device) -> object:
    for name, value in vars(policy).items():
        if isinstance(value, torch.Tensor):
            setattr(policy, name, value.to(device))
    return policy


def env_worker(
    env_id: int,
    num_agents: int,
    obs_queue: mp.Queue,
    action_queue: mp.Queue,
) -> None:
    env = MultiAgentDummyEnv(env_id, num_agents)
    obs = env.reset()
    while True:
        obs_queue.put((env_id, {k: v.tolist() for k, v in obs.items()}))
        action = action_queue.get()
        if action is None:
            break
        action_tensor = torch.tensor(action, dtype=torch.float32)
        obs, _, _, _ = env.step(action_tensor)


def run_parallel_demo(
    num_envs: int,
    num_agents: int,
    rollout_steps: int,
    device: torch.device,
) -> None:
    obs_queue: mp.Queue = mp.Queue()
    action_queues: Dict[int, mp.Queue] = {}
    workers: List[mp.Process] = []

    for env_id in range(num_envs):
        action_queue: mp.Queue = mp.Queue()
        worker = mp.Process(
            target=env_worker,
            args=(env_id, num_agents, obs_queue, action_queue),
            daemon=True,
        )
        worker.start()
        workers.append(worker)
        action_queues[env_id] = action_queue

    policies = [policy_to_device(build_policy(agent_id), device) for agent_id in range(num_agents)]
    print("Assigned policies:")
    for agent_id, policy in enumerate(policies):
        print(f"  Agent {agent_id}: {policy.__class__.__name__}")

    try:
        last_ts = time.perf_counter()
        total_steps = 0
        total_time = 0.0
        for step in range(rollout_steps):
            env_obs: Dict[int, torch.Tensor] = {}
            for _ in range(num_envs):
                env_id, obs_dict = obs_queue.get()
                env_obs[env_id] = {
                    key: torch.tensor(value, dtype=torch.float32) for key, value in obs_dict.items()
                }

            env_ids = sorted(env_obs)
            stacked_obs = {}
            for key in env_obs[env_ids[0]].keys():
                stacked_obs[key] = torch.stack(
                    [
                        torch.stack([env_obs[env_id][key][agent_id] for env_id in env_ids])
                        for agent_id in range(num_agents)
                    ],
                    dim=0,
                )

            state_obs = stacked_obs["state"].to(device)
            agents_obs = state_obs.permute(0, 2, 1)
            print(f"Step {step}: agents_obs shape {agents_obs.shape}")

            per_agent_actions: List[torch.Tensor] = []
            for agent_id, policy in enumerate(policies):
                obs_for_agent = state_obs[agent_id]
                actions = policy(obs_for_agent)
                per_agent_actions.append(actions)

            batched_actions = torch.stack(per_agent_actions)
            batched_actions_cpu = batched_actions.cpu()

            for env_index, env_id in enumerate(env_ids):
                action = batched_actions_cpu[:, env_index, :]
                action_queues[env_id].put(action.tolist())

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
    finally:
        for q in action_queues.values():
            q.put(None)
        for worker in workers:
            worker.join()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel env demo")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-agents", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=1000)
    parser.add_argument(
        "--device",
        choices=["cpu", "mps"],
        default="cpu",
        help="Device to run batched policies on",
    )
    return parser.parse_args()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    requested_device = args.device
    if requested_device == "mps" and not torch.backends.mps.is_available():
        print("MPS requested but not available; falling back to CPU")
        requested_device = "cpu"
    device = torch.device(requested_device)
    run_parallel_demo(
        num_envs=args.num_envs,
        num_agents=args.num_agents,
        rollout_steps=args.rollout_steps,
        device=device,
    )
