"""PettingZoo Tennis environment wrapper for Simverse."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional

from gymnasium import spaces


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


__all__ = ["PettingZooTennisEnv"]
