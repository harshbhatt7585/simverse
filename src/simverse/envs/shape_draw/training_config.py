from __future__ import annotations

from typing import Any, Dict, Optional

import torch


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    resolved_buffer_size = buffer_size if buffer_size is not None else max_steps * num_agents * 32
    resolved_batch_size = (
        batch_size
        if batch_size is not None
        else min(4096, max(256, resolved_buffer_size // max(num_agents * 8, 1)))
    )
    if resolved_device == "mps":
        resolved_batch_size = min(resolved_batch_size, 1024)

    return {
        "num_agents": num_agents,
        "num_envs": max(1, int(num_envs)),
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
        "tensor_buffer_max_capacity": 2048,
    }
