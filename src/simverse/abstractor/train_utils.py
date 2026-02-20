from __future__ import annotations

import inspect
from typing import Any, Mapping, Sequence

import torch

from simverse.config.policy import PolicySpec


def resolve_torch_device(*, prefer_mps: bool = True) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_rollout_dtype(
    device: str,
    *,
    cuda_dtype: torch.dtype = torch.float16,
    mps_dtype: torch.dtype = torch.bfloat16,
    cpu_dtype: torch.dtype = torch.float32,
) -> torch.dtype:
    if device == "cuda":
        return cuda_dtype
    if device == "mps":
        return mps_dtype
    return cpu_dtype


def configure_torch_backend(device: str) -> None:
    if device != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def compile_policy_models(
    policy_specs: Sequence[PolicySpec],
    *,
    use_compile: bool,
    device: str,
) -> list[torch.nn.Module]:
    models = [policy.model for policy in policy_specs]
    if use_compile and hasattr(torch, "compile") and device == "cuda":
        return [torch.compile(model, mode="max-autotune") for model in models]
    return models


def build_adam_optimizers(
    policy_models: Sequence[torch.nn.Module],
    *,
    lr: float,
    device: str,
) -> dict[int, torch.optim.Optimizer]:
    adam_kwargs: dict[str, Any] = {}
    if "fused" in inspect.signature(torch.optim.Adam).parameters and device == "cuda":
        adam_kwargs["fused"] = True
    return {
        agent_id: torch.optim.Adam(model.parameters(), lr=lr, **adam_kwargs)
        for agent_id, model in enumerate(policy_models)
    }


def build_ppo_training_config(
    *,
    num_agents: int,
    num_envs: int,
    max_steps: int,
    episodes: int,
    training_epochs: int,
    lr: float,
    batch_size: int,
    buffer_size: int,
    device: str,
    dtype: torch.dtype,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = {
        "num_agents": int(num_agents),
        "num_envs": int(num_envs),
        "max_steps": int(max_steps),
        "episodes": int(episodes),
        "training_epochs": int(training_epochs),
        "clip_epsilon": 0.2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "lr": float(lr),
        "batch_size": int(batch_size),
        "buffer_size": int(buffer_size),
        "device": device,
        "dtype": dtype,
        "torch_fastpath": True,
    }
    if extras:
        config.update(dict(extras))
    return config
