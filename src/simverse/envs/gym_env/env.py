from __future__ import annotations

import torch

from simverse.envs.gym_env.torch_env import GymTorchConfig, GymTorchEnv


class GymEnv(GymTorchEnv):
    """Canonical Gym environment entrypoint."""


def create_env(
    config: GymTorchConfig,
    *,
    num_envs: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> GymEnv:
    return GymEnv(config=config, num_envs=num_envs, device=device, dtype=dtype)
