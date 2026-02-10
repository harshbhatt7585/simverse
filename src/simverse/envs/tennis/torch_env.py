from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Tuple

import numpy as np
import torch

from simverse.abstractor.simtorch_env import SimTorchEnv
from simverse.envs.tennis.agent import TennisAgent
from simverse.envs.tennis.config import TennisConfig
from simverse.envs.tennis.env import TennisEnv


class TennisTorchEnv(SimTorchEnv):
    """Batched tennis environment returning torch tensors for PPO torch fastpath."""

    def __init__(
        self,
        config: TennisConfig,
        num_envs: int | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.config = config
        if self.config.num_agents != 2:
            raise ValueError("TennisTorchEnv currently supports exactly 2 agents")

        self.num_envs = num_envs or getattr(config, "num_envs", 1)
        self.num_agents = self.config.num_agents
        self.envs: list[TennisEnv] = [TennisEnv(deepcopy(config)) for _ in range(self.num_envs)]
        self.agents: list[TennisAgent] = []
        self.to(self.device)

    @property
    def action_space(self):
        return self.envs[0].action_space

    @property
    def observation_space(self):
        return self.envs[0].observation_space

    @property
    def local_observation_space(self):
        return self.envs[0].local_observation_space

    def assign_agents(self, agents: list[TennisAgent]) -> None:
        if len(agents) != self.config.num_agents:
            raise ValueError(f"Tennis requires {self.config.num_agents} agents")
        self.agents = agents

        templates = {agent.agent_id: agent for agent in agents}
        for env in self.envs:
            env_agents: list[TennisAgent] = []
            for agent_id in range(self.num_agents):
                template = templates.get(agent_id)
                policy = template.policy if template is not None else None
                action_space = getattr(template, "action_space", None)
                if action_space is None:
                    action_count = int(getattr(env.action_space, "n", 18))
                    action_space = torch.arange(action_count, dtype=torch.int64).cpu().numpy()
                env_agents.append(
                    TennisAgent(
                        agent_id=agent_id,
                        action_space=action_space,
                        policy=policy,
                        name=f"tennis_agent_{agent_id}",
                    )
                )
            env.assign_agents(env_agents)

    def reset(self) -> Dict[str, torch.Tensor]:
        observations = [env.reset() for env in self.envs]
        return self._stack_observation_payload(observations)

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        action_tensor = self._normalize_actions(actions)
        rewards = torch.zeros(
            (self.num_envs, self.num_agents), dtype=self.dtype, device=self.device
        )

        obs_payloads: list[Dict[str, Any]] = []
        info_list: list[Dict[str, Any]] = []
        done_flags = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        action_cpu = action_tensor.detach().cpu().numpy()
        for env_idx, env in enumerate(self.envs):
            env_actions: Dict[int, int] = {}
            for agent_id in range(self.num_agents):
                action = int(action_cpu[env_idx, agent_id])
                if action >= 0:
                    env_actions[agent_id] = action

            obs, reward_dict, done, info = env.step(env_actions)
            obs_payloads.append(obs)
            info_list.append(info if isinstance(info, dict) else {})
            done_flags[env_idx] = bool(done)

            for agent_id in range(self.num_agents):
                rewards[env_idx, agent_id] = float(reward_dict.get(agent_id, 0.0))

        stacked_obs = self._stack_observation_payload(obs_payloads)
        info: Dict[str, Any] = {
            "winner": stacked_obs["winner"].clone(),
            "steps": stacked_obs["steps"].clone(),
            "infos": info_list,
        }
        return stacked_obs, rewards, done_flags, info

    def _normalize_actions(self, actions: torch.Tensor | None) -> torch.Tensor:
        if actions is None:
            return torch.full(
                (self.num_envs, self.num_agents), -1, dtype=torch.int64, device=self.device
            )
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        if actions.ndim == 1:
            if self.num_envs == 1 and actions.numel() == self.num_agents:
                actions = actions.unsqueeze(0)
            elif self.num_agents == 1 and actions.numel() == self.num_envs:
                actions = actions.unsqueeze(1)
        if actions.shape != (self.num_envs, self.num_agents):
            expected_shape = (self.num_envs, self.num_agents)
            actual_shape = tuple(actions.shape)
            raise ValueError(f"Expected actions shape {expected_shape}, got {actual_shape}")
        if actions.device != self.device or actions.dtype != torch.int64:
            actions = actions.to(device=self.device, dtype=torch.int64)
        return actions

    def _stack_observation_payload(
        self, observations: list[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        obs_array = np.stack([obs["obs"] for obs in observations], axis=0)
        local_obs_array = np.stack([obs["local_obs"] for obs in observations], axis=0)
        obs_tensor = torch.from_numpy(obs_array).to(device=self.device, dtype=self.dtype)
        local_obs_tensor = torch.from_numpy(local_obs_array).to(
            device=self.device,
            dtype=self.dtype,
        )
        done = torch.as_tensor(
            [bool(obs.get("done", False)) for obs in observations],
            dtype=torch.bool,
            device=self.device,
        )
        winner = torch.as_tensor(
            [
                -1 if obs.get("winner", None) is None else int(obs.get("winner", -1))
                for obs in observations
            ],
            dtype=torch.int64,
            device=self.device,
        )
        steps = torch.as_tensor(
            [int(obs.get("steps", 0)) for obs in observations],
            dtype=torch.int64,
            device=self.device,
        )
        return {
            "obs": obs_tensor,
            "local_obs": local_obs_tensor,
            "done": done,
            "winner": winner,
            "steps": steps,
        }
