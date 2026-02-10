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
    width: int = 20,
    height: int = 20,
    num_agents: int = 4,
    num_envs: int = 256,
    max_steps: int = 1000,
    episodes: int = 100,
    training_epochs: int = 1,
    lr: float = 0.001,
    clip_epsilon: float = 0.2,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    total_seeds: int = 500,
    batch_size: Optional[int] = None,
    buffer_size: Optional[int] = None,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    resolved_device = device or select_device()
    resolved_buffer_size = (
        buffer_size if buffer_size is not None else num_envs * num_agents * 10
    )
    resolved_batch_size = (
        batch_size
        if batch_size is not None
        else min(8192, resolved_buffer_size // max(num_agents, 1))
    )

    return {
        "width": width,
        "height": height,
        "num_agents": num_agents,
        "num_envs": num_envs,
        "max_steps": max_steps,
        "episodes": episodes,
        "training_epochs": training_epochs,
        "lr": lr,
        "clip_epsilon": clip_epsilon,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "total_seeds": total_seeds,
        "batch_size": resolved_batch_size,
        "buffer_size": resolved_buffer_size,
        "device": resolved_device,
        "dtype": dtype,
    }
