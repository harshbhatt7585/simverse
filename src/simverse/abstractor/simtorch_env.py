from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn


class SimTorchEnv(nn.Module, ABC):
    """Base class for torch-native, batched simulation environments."""

    def __init__(self, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype

    @property
    @abstractmethod
    def action_space(self) -> Any: ...

    @property
    @abstractmethod
    def observation_space(self) -> Any: ...

    @abstractmethod
    def reset(self) -> Dict[str, torch.Tensor]: ...

    @abstractmethod
    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]: ...

    def get_observation(self) -> Dict[str, torch.Tensor]:
        return self.reset()

    def _resolve_num_envs(self, num_envs: int | None, config: Any, *, default: int = 1) -> int:
        resolved = num_envs
        if resolved is None:
            resolved = getattr(config, "num_envs", default)
        return max(1, int(resolved))

    def _normalize_action_matrix(
        self,
        actions: torch.Tensor | Sequence[int] | None,
        *,
        num_agents: int | None = None,
        missing_action: int = -1,
    ) -> torch.Tensor:
        env_count = int(self.num_envs)
        agent_count = int(self.num_agents if num_agents is None else num_agents)

        if actions is None:
            return torch.full(
                (env_count, agent_count),
                int(missing_action),
                dtype=torch.int64,
                device=self.device,
            )

        action_tensor = actions if isinstance(actions, torch.Tensor) else torch.as_tensor(actions)
        if action_tensor.ndim == 1:
            if env_count == 1 and action_tensor.numel() == agent_count:
                action_tensor = action_tensor.unsqueeze(0)
            elif agent_count == 1 and action_tensor.numel() == env_count:
                action_tensor = action_tensor.unsqueeze(1)

        expected_shape = (env_count, agent_count)
        if tuple(action_tensor.shape) != expected_shape:
            raise ValueError(
                f"Expected actions shape {expected_shape}, got {tuple(action_tensor.shape)}"
            )
        return action_tensor.to(device=self.device, dtype=torch.int64)

    def _normalize_single_agent_actions(
        self,
        actions: torch.Tensor | Sequence[int] | Mapping[int, int] | None,
        *,
        missing_action: int = -1,
        dict_default: int | None = None,
    ) -> torch.Tensor:
        env_count = int(self.num_envs)

        if actions is None:
            return torch.full(
                (env_count,),
                int(missing_action),
                dtype=torch.int64,
                device=self.device,
            )

        if isinstance(actions, Mapping):
            if env_count != 1:
                raise ValueError("Dict actions are supported only when num_envs == 1")
            fallback = missing_action if dict_default is None else dict_default
            return torch.as_tensor(
                [int(actions.get(0, fallback))],
                dtype=torch.int64,
                device=self.device,
            )

        action_tensor = actions if isinstance(actions, torch.Tensor) else torch.as_tensor(actions)
        if action_tensor.ndim == 0:
            action_tensor = action_tensor.unsqueeze(0)
        elif action_tensor.ndim == 2 and action_tensor.shape[1] == 1:
            action_tensor = action_tensor[:, 0]
        if action_tensor.ndim != 1:
            raise ValueError(
                "Expected actions with shape [num_envs] or [num_envs, 1], "
                f"got {tuple(action_tensor.shape)}"
            )

        if action_tensor.shape[0] == 1 and env_count > 1:
            action_tensor = action_tensor.repeat(env_count)

        if action_tensor.shape[0] != env_count:
            raise ValueError(f"Expected {env_count} actions, got {int(action_tensor.shape[0])}")

        return action_tensor.to(device=self.device, dtype=torch.int64)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if args:
            self.device = torch.device(args[0])
        elif "device" in kwargs:
            self.device = torch.device(kwargs["device"])
        return self
