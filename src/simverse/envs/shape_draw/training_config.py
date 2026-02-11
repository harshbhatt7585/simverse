from __future__ import annotations

from typing import Any, Dict, Optional

import torch


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _derive_batch_size(
    *,
    num_envs: int,
    requested_batch_size: Optional[int],
    device: str,
) -> int:
    batch_size = int(requested_batch_size) if requested_batch_size is not None else num_envs * 2
    batch_size = _round_up_to_multiple(max(num_envs, batch_size), num_envs)
    if device == "mps" and num_envs <= 1024:
        batch_size = min(batch_size, 1024)
        batch_size = _round_up_to_multiple(max(num_envs, batch_size), num_envs)
    return batch_size


def _derive_buffer_size(
    *,
    num_envs: int,
    num_agents: int,
    batch_size: int,
    requested_buffer_size: Optional[int],
) -> int:
    min_buffer_size = batch_size * num_agents
    default_buffer_size = min_buffer_size * 4
    buffer_size = (
        int(requested_buffer_size) if requested_buffer_size is not None else default_buffer_size
    )
    return _round_up_to_multiple(max(min_buffer_size, buffer_size), num_envs * num_agents)


def build_training_config(
    *,
    num_agents: int = 1,
    num_envs: int = 64,
    max_steps: int = 256,
    episodes: int = 200,
    training_epochs: int = 1,
    lr: float = 3e-4,
    clip_epsilon: float = 0.2,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    entropy_coef: float = 0.01,
    normalize_advantages: bool = True,
    batch_size: Optional[int] = None,
    buffer_size: Optional[int] = None,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    resolved_device = device or select_device()
    resolved_num_envs = max(1, int(num_envs))
    resolved_num_agents = max(1, int(num_agents))
    resolved_batch_size = _derive_batch_size(
        num_envs=resolved_num_envs,
        requested_batch_size=batch_size,
        device=resolved_device,
    )
    resolved_buffer_size = _derive_buffer_size(
        num_envs=resolved_num_envs,
        num_agents=resolved_num_agents,
        batch_size=resolved_batch_size,
        requested_buffer_size=buffer_size,
    )
    tensor_buffer_max_capacity = resolved_buffer_size // resolved_num_agents

    return {
        "num_agents": resolved_num_agents,
        "num_envs": resolved_num_envs,
        "max_steps": max_steps,
        "episodes": episodes,
        "training_epochs": training_epochs,
        "lr": lr,
        "clip_epsilon": clip_epsilon,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "entropy_coef": entropy_coef,
        "normalize_advantages": normalize_advantages,
        "batch_size": resolved_batch_size,
        "buffer_size": resolved_buffer_size,
        "device": resolved_device,
        "dtype": dtype,
        "torch_fastpath": True,
        "tensor_buffer_max_capacity": tensor_buffer_max_capacity,
    }
