from __future__ import annotations

import torch

from simverse.envs.battle_grid.config import BattleGridConfig
from simverse.envs.battle_grid.torch_env import BattleGridTorchEnv


class BattleGridEnv(BattleGridTorchEnv):
    """Canonical Battle Grid environment entrypoint."""


def create_env(
    config: BattleGridConfig,
    *,
    num_envs: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> BattleGridEnv:
    return BattleGridEnv(config=config, num_envs=num_envs, device=device, dtype=dtype)
