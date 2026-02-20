from __future__ import annotations

import torch

from simverse.envs.maze_race.config import MazeRaceConfig
from simverse.envs.maze_race.torch_env import MazeRaceTorchEnv


class MazeRaceEnv(MazeRaceTorchEnv):
    """Canonical Maze Race environment entrypoint."""


def create_env(
    config: MazeRaceConfig,
    *,
    num_envs: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> MazeRaceEnv:
    return MazeRaceEnv(config=config, num_envs=num_envs, device=device, dtype=dtype)
