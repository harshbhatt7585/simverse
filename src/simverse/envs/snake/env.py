from __future__ import annotations

import torch

from simverse.envs.snake.config import SnakeConfig
from simverse.envs.snake.torch_env import SnakeTorchEnv


class SnakeEnv(SnakeTorchEnv):
    """Canonical Snake environment entrypoint."""


def create_env(
    config: SnakeConfig,
    *,
    num_envs: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> SnakeEnv:
    return SnakeEnv(config=config, num_envs=num_envs, device=device, dtype=dtype)
