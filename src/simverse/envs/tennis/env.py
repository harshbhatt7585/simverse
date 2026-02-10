"""PettingZoo Tennis environment wrappers for Simverse."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simverse.abstractor.simenv import SimEnv
from simverse.abstractor.simvector_env import SimVectorEnv
from simverse.envs.tennis.agent import TennisAgent
from simverse.envs.tennis.config import TennisConfig


def _resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
    if frame.shape[0] == size and frame.shape[1] == size:
        return frame
    # Nearest-neighbor downsample without extra dependencies.
    y_idx = np.linspace(0, frame.shape[0] - 1, size).astype(np.int32)
    x_idx = np.linspace(0, frame.shape[1] - 1, size).astype(np.int32)
    return frame[np.ix_(y_idx, x_idx)]


def _process_observation(raw_obs: np.ndarray, *, resize: int, grayscale: bool) -> np.ndarray:
    frame = np.asarray(raw_obs)
    if frame.ndim != 3:
        raise ValueError(f"Unexpected tennis observation shape: {frame.shape}")

    frame = _resize_frame(frame, resize)
    if grayscale:
        gray = (0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]).astype(
            np.float32
        )
        return np.expand_dims(gray / 255.0, axis=0)
    return np.transpose(frame.astype(np.float32) / 255.0, (2, 0, 1))


class PettingZooTennisEnv:
    """Thin wrapper around PettingZoo Atari tennis parallel environment."""

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_cycles: int = 900,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        try:
            from pettingzoo.atari import tennis_v3
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PettingZoo tennis env requires optional dependencies. "
                "Install with: pip install -e .[pettingzoo]"
            ) from exc

        if "rom_path" in kwargs and "auto_rom_install_path" not in kwargs:
            rom_path = Path(str(kwargs.pop("rom_path"))).expanduser()
            kwargs["auto_rom_install_path"] = str(
                rom_path.parent if rom_path.suffix == ".bin" else rom_path
            )

        self._env = tennis_v3.parallel_env(
            render_mode=render_mode,
            max_cycles=max_cycles,
            **kwargs,
        )
        self.possible_agents = list(self._env.possible_agents)
        self._seed = seed

    @property
    def agents(self) -> list[str]:
        return list(self._env.agents)

    @property
    def action_space(self) -> spaces.Dict:
        return spaces.Dict({agent: self._env.action_space(agent) for agent in self.possible_agents})

    @property
    def observation_space(self) -> spaces.Dict:
        return spaces.Dict(
            {agent: self._env.observation_space(agent) for agent in self.possible_agents}
        )

    def reset(self, seed: Optional[int] = None) -> tuple[Dict[str, Any], Dict[str, Any]]:
        reset_seed = self._seed if seed is None else seed
        result = self._env.reset(seed=reset_seed)
        if isinstance(result, tuple):
            observations, infos = result
        else:
            observations = result
            infos = {agent: {} for agent in self._env.agents}
        return observations, infos

    def step(
        self, actions: Mapping[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        return self._env.step(dict(actions))

    def render(self) -> Any:
        return self._env.render()

    def close(self) -> None:
        self._env.close()


class TennisEnv(SimEnv):
    """Simverse-compatible adapter over PettingZoo Atari tennis."""

    def __init__(self, config: TennisConfig):
        self.config = config
        if self.config.num_agents != 2:
            raise ValueError("TennisEnv currently supports exactly 2 agents")

        self._raw_env = PettingZooTennisEnv(
            render_mode=None,
            max_cycles=self.config.max_steps,
            seed=self.config.seed,
            obs_type="rgb_image",
        )
        self._agent_names = list(self._raw_env.possible_agents)
        self.agents: list[TennisAgent] = []
        self.steps = 0
        self.done = False
        self.winner: int | None = None
        self._episode_rewards = np.zeros(self.config.num_agents, dtype=np.float32)

        sample_space = self._raw_env._env.observation_space(self._agent_names[0])
        channels = 1 if self.config.use_grayscale else int(sample_space.shape[2])
        self._single_agent_obs_shape = (
            channels,
            self.config.obs_resize,
            self.config.obs_resize,
        )
        self._obs_shape = (
            channels * self.config.num_agents,
            self.config.obs_resize,
            self.config.obs_resize,
        )
        self._observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=self._obs_shape,
            dtype=np.float32,
        )
        self._local_observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=self._single_agent_obs_shape,
            dtype=np.float32,
        )
        action_n = int(self._raw_env._env.action_space(self._agent_names[0]).n)
        self._action_space = gym.spaces.Discrete(action_n)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def local_observation_space(self):
        return self._local_observation_space

    def assign_agents(self, agents: list[TennisAgent]) -> None:
        if len(agents) != self.config.num_agents:
            raise ValueError(f"Tennis requires {self.config.num_agents} agents")
        self.agents = agents

    def _normalize_actions(
        self, actions: Dict[int, int] | Iterable[int] | int | None
    ) -> Dict[int, int]:
        if actions is None:
            return {}
        if isinstance(actions, dict):
            return {int(agent_id): int(action) for agent_id, action in actions.items()}
        if isinstance(actions, (list, tuple)) and actions and isinstance(actions[0], dict):
            merged: Dict[int, int] = {}
            for batch_actions in actions:
                for agent_id, action in batch_actions.items():
                    merged[int(agent_id)] = int(action)
            return merged
        if isinstance(actions, int):
            return {0: int(actions)}
        return {int(agent_id): int(action) for agent_id, action in enumerate(actions)}

    def _build_observation_payload(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        stacked: list[np.ndarray] = []
        agent_info: list[dict[str, Any]] = []
        for agent in self.agents:
            agent_name = self._agent_names[agent.agent_id]
            raw_obs = observations.get(agent_name)
            if raw_obs is None:
                proc = np.zeros(self._single_agent_obs_shape, dtype=np.float32)
            else:
                proc = _process_observation(
                    raw_obs,
                    resize=self.config.obs_resize,
                    grayscale=self.config.use_grayscale,
                )
            stacked.append(proc)
            agent_info.append(agent.info())

        if stacked:
            local_obs = np.stack(stacked, axis=0)
            obs_tensor = np.concatenate(stacked, axis=0)
        else:
            local_obs = np.zeros(
                (self.config.num_agents, *self._single_agent_obs_shape),
                dtype=np.float32,
            )
            obs_tensor = np.zeros(self._obs_shape, dtype=np.float32)
        return {
            "obs": obs_tensor,
            "local_obs": local_obs,
            "agents": agent_info,
            "done": self.done,
            "winner": self.winner,
            "steps": self.steps,
        }

    def _update_winner(self) -> None:
        if self._episode_rewards[0] > self._episode_rewards[1]:
            self.winner = 0
        elif self._episode_rewards[1] > self._episode_rewards[0]:
            self.winner = 1
        else:
            self.winner = None

    def reset(self):
        observations, _infos = self._raw_env.reset(seed=self.config.seed)
        self.steps = 0
        self.done = False
        self.winner = None
        self._episode_rewards.fill(0.0)
        for agent in self.agents:
            agent.reset()
        return self._build_observation_payload(observations)

    def step(self, actions: Dict[int, int] | Iterable[int] | int | None = None):
        if self.done:
            info = {"winner": self.winner, "steps": self.steps}
            rewards = {agent.agent_id: 0.0 for agent in self.agents}
            return self.get_observation(), rewards, True, info

        action_map = self._normalize_actions(actions)
        alive_names = self._raw_env.agents
        pz_actions: Dict[str, int] = {}
        for name in alive_names:
            agent_id = self._agent_names.index(name)
            pz_actions[name] = int(action_map.get(agent_id, 0))

        observations, rewards_raw, terminations, truncations, infos = self._raw_env.step(pz_actions)
        self.steps += 1

        reward_dict: Dict[int, float] = {}
        for agent_id, agent_name in enumerate(self._agent_names):
            reward = float(rewards_raw.get(agent_name, 0.0))
            reward_dict[agent_id] = reward
            if agent_id < len(self.agents):
                self.agents[agent_id].reward = reward
            self._episode_rewards[agent_id] += reward

        terminated = bool(terminations) and all(bool(v) for v in terminations.values())
        truncated = bool(truncations) and all(bool(v) for v in truncations.values())
        self.done = terminated or truncated or self.steps >= self.config.max_steps
        if self.done:
            self._update_winner()

        info = {
            "winner": self.winner,
            "steps": self.steps,
            "terminations": terminations,
            "truncations": truncations,
            "infos": infos,
        }
        return self._build_observation_payload(observations), reward_dict, self.done, info

    def render(self) -> Any:
        return self._raw_env.render()

    def close(self) -> None:
        self._raw_env.close()

    def get_observation(self) -> Dict[str, Any]:
        observations = {}
        for name in self._agent_names:
            observations[name] = self._raw_env._env.observe(name)
        return self._build_observation_payload(observations)


class TennisVectorizedEnv(SimVectorEnv):
    """Runs multiple independent TennisEnv copies in lockstep."""

    def __init__(self, config: TennisConfig, num_envs: int | None = None) -> None:
        self.config = config
        resolved_envs = num_envs or getattr(config, "num_envs", 1)
        super().__init__(resolved_envs)

    def _create_env(self, index: int) -> SimEnv:
        return TennisEnv(deepcopy(self.config))

    def _stack_rewards(self, reward_dicts: list[Dict[int, float]]) -> np.ndarray:
        reward_array = np.zeros((self.num_envs, self.config.num_agents), dtype=np.float32)
        for env_idx, rewards in enumerate(reward_dicts):
            for agent_id in range(self.config.num_agents):
                reward_array[env_idx, agent_id] = float(rewards.get(agent_id, 0.0))
        return reward_array

    def _stack_observations(self, observations: list[Dict[str, Any]]) -> Dict[str, Any]:
        obs_tensor = np.stack([obs["obs"] for obs in observations], axis=0)
        local_obs = np.stack([obs["local_obs"] for obs in observations], axis=0)
        steps = np.asarray([obs.get("steps", 0) for obs in observations], dtype=np.int32)
        done_flags = np.asarray([obs.get("done", False) for obs in observations], dtype=np.bool_)
        return {
            "obs": obs_tensor,
            "local_obs": local_obs,
            "agents": [obs.get("agents", []) for obs in observations],
            "done": done_flags,
            "winner": [obs.get("winner") for obs in observations],
            "steps": steps,
        }

    def assign_agents(self, agents: list[TennisAgent]) -> None:
        if len(agents) != self.config.num_agents:
            raise ValueError(f"Tennis requires {self.config.num_agents} agents")
        self.agents = agents

        templates = {agent.agent_id: agent for agent in agents}
        for env in self.envs:
            env_agents: list[TennisAgent] = []
            for agent_id in range(self.config.num_agents):
                template = templates.get(agent_id)
                policy = template.policy if template is not None else None
                action_space = getattr(template, "action_space", None)
                if not isinstance(action_space, np.ndarray):
                    action_count = int(getattr(env.action_space, "n", 18))
                    action_space = np.arange(action_count, dtype=np.int64)
                env_agents.append(
                    TennisAgent(
                        agent_id=agent_id,
                        action_space=action_space,
                        policy=policy,
                        name=f"tennis_agent_{agent_id}",
                    )
                )
            env.assign_agents(env_agents)


__all__ = ["PettingZooTennisEnv", "TennisEnv", "TennisVectorizedEnv", "TennisConfig"]
