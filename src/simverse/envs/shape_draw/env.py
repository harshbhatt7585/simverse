from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List

import gymnasium as gym
import numpy as np

from simverse.abstractor.simenv import SimEnv
from simverse.abstractor.simvector_env import SimVectorEnv
from simverse.envs.shape_draw.agent import ShapeDrawAgent
from simverse.envs.shape_draw.config import ShapeDrawConfig


class ShapeDrawEnv(SimEnv):
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_TOGGLE_PEN = 4
    ACTION_BRUSH_UP = 5
    ACTION_BRUSH_DOWN = 6
    ACTION_SPACE = gym.spaces.Discrete(7)

    @property
    def action_space(self):
        return self.ACTION_SPACE

    @property
    def observation_space(self):
        # canvas, target, pen marker
        return gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, self.config.height, self.config.width),
            dtype=np.float32,
        )

    def __init__(self, config: ShapeDrawConfig):
        self.config = config
        if self.config.num_agents != 1:
            raise ValueError("ShapeDrawEnv currently supports exactly 1 agent")

        self.width = self.config.width
        self.height = self.config.height
        self.rng = np.random.default_rng(self.config.seed)

        self.canvas = np.zeros((1, self.height, self.width), dtype=np.float32)
        self.target = np.zeros((1, self.height, self.width), dtype=np.float32)
        self.target_mask = np.zeros((self.height, self.width), dtype=np.bool_)
        self.target_pixels = 1
        self.pen_x = 0
        self.pen_y = 0
        self.pen_down = False
        self.brush = 2
        self.steps = 0
        self.done = False
        self.winner: int | None = None
        self.similarity = 0.0

        self.agents: list[ShapeDrawAgent] = []

    def assign_agents(self, agents: list[ShapeDrawAgent]) -> None:
        if len(agents) != 1:
            raise ValueError("ShapeDraw requires exactly 1 agent")
        self.agents = agents

    def reset(self):
        self.canvas.fill(0.0)
        self.target = self._sample_target()
        self._cache_target_mask()
        self.pen_x = int(self.rng.integers(0, self.width))
        self.pen_y = int(self.rng.integers(0, self.height))
        self.pen_down = False
        self.brush = 2
        self.steps = 0
        self.done = False
        self.winner = None
        self.similarity = self._similarity()
        for agent in self.agents:
            agent.reset()
        return self._observation_payload()

    def step(self, actions: Dict[int, int] | Iterable[int] | int | None = None):
        if self.done:
            reward = {0: 0.0}
            info = {
                "similarity": float(self.similarity),
                "pen_down": bool(self.pen_down),
                "brush": int(self.brush),
                "steps": int(self.steps),
            }
            return self._observation_payload(), reward, True, info

        action = self._normalize_action(actions)
        prev_canvas_mask = self.canvas[0] > 0.5

        if action == self.ACTION_UP:
            self.pen_y -= 1
        elif action == self.ACTION_DOWN:
            self.pen_y += 1
        elif action == self.ACTION_LEFT:
            self.pen_x -= 1
        elif action == self.ACTION_RIGHT:
            self.pen_x += 1
        elif action == self.ACTION_TOGGLE_PEN:
            self.pen_down = not self.pen_down
        elif action == self.ACTION_BRUSH_UP:
            self.brush += 1
        elif action == self.ACTION_BRUSH_DOWN:
            self.brush -= 1

        self.pen_x = int(np.clip(self.pen_x, 0, self.width - 1))
        self.pen_y = int(np.clip(self.pen_y, 0, self.height - 1))
        self.brush = int(np.clip(self.brush, self.config.min_brush, self.config.max_brush))

        if self.pen_down:
            self._draw_disk()

        self.steps += 1
        self.similarity = self._similarity()

        canvas_mask = self.canvas[0] > 0.5
        newly_filled = np.logical_and(canvas_mask, np.logical_not(prev_canvas_mask))
        new_target_fills = np.logical_and(newly_filled, self.target_mask)
        new_wrong_fills = np.logical_and(newly_filled, np.logical_not(self.target_mask))
        reward = float(new_target_fills.sum()) / float(self.target_pixels)
        reward -= float(self.config.wrong_draw_penalty) * (
            float(new_wrong_fills.sum()) / float(self.target_pixels)
        )
        reward -= float(self.config.step_penalty)
        if self.pen_down:
            reward -= float(self.config.draw_penalty)

        solved = self.similarity >= float(self.config.completion_threshold)
        timed_out = self.steps >= self.config.max_steps
        self.done = bool(solved or timed_out)
        if solved:
            reward += float(self.config.completion_bonus)

        if self.agents:
            self.agents[0].reward = reward

        reward_dict = {0: reward}
        info = {
            "similarity": float(self.similarity),
            "pen_down": bool(self.pen_down),
            "brush": int(self.brush),
            "steps": int(self.steps),
        }
        return self._observation_payload(), reward_dict, self.done, info

    def get_observation(self):
        return self._observation_payload()

    def _normalize_action(self, actions: Dict[int, int] | Iterable[int] | int | None) -> int:
        if actions is None:
            return -1
        if isinstance(actions, int):
            return int(actions)
        if isinstance(actions, dict):
            return int(actions.get(0, -1))
        if isinstance(actions, (list, tuple)):
            if not actions:
                return -1
            first = actions[0]
            if isinstance(first, dict):
                return int(first.get(0, -1))
            return int(first)
        return int(actions)

    def _sample_target(self) -> np.ndarray:
        target = np.zeros((1, self.height, self.width), dtype=np.float32)
        cx = self.width // 2
        cy = self.height // 2
        size = int(self.rng.integers(max(6, self.width // 8), max(7, self.width // 4)))
        mask = self._circle_mask(cx, cy, size)
        target[0, mask] = 1.0
        return target

    def _cache_target_mask(self) -> None:
        self.target_mask = self.target[0] > 0.5
        target_pixels = int(self.target_mask.sum())
        self.target_pixels = max(target_pixels, 1)

    def _circle_mask(self, cx: int, cy: int, radius: int) -> np.ndarray:
        yy, xx = np.ogrid[: self.height, : self.width]
        return (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy) <= radius * radius

    def _square_mask(self, cx: int, cy: int, half: int) -> np.ndarray:
        x0 = max(0, cx - half)
        x1 = min(self.width, cx + half)
        y0 = max(0, cy - half)
        y1 = min(self.height, cy + half)
        mask = np.zeros((self.height, self.width), dtype=np.bool_)
        mask[y0:y1, x0:x1] = True
        return mask

    def _triangle_mask(self, cx: int, cy: int, size: int) -> np.ndarray:
        top = max(0, cy - size)
        bottom = min(self.height - 1, cy + size)
        span = max(bottom - top, 1)
        mask = np.zeros((self.height, self.width), dtype=np.bool_)
        for y in range(top, bottom + 1):
            frac = (y - top) / span
            half_width = int(size * frac)
            x0 = max(0, cx - half_width)
            x1 = min(self.width, cx + half_width + 1)
            mask[y, x0:x1] = True
        return mask

    def _draw_disk(self) -> None:
        r = int(self.brush)
        x0 = max(0, self.pen_x - r)
        x1 = min(self.width - 1, self.pen_x + r)
        y0 = max(0, self.pen_y - r)
        y1 = min(self.height - 1, self.pen_y + r)
        ys, xs = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
        disk = (
            (xs - self.pen_x) * (xs - self.pen_x) + (ys - self.pen_y) * (ys - self.pen_y)
        ) <= r * r
        patch = self.canvas[0, y0 : y1 + 1, x0 : x1 + 1]
        patch[disk] = 1.0

    def _similarity(self) -> float:
        target_mask = self.target > 0.5
        canvas_mask = self.canvas > 0.5
        intersection = float(np.logical_and(target_mask, canvas_mask).sum())
        union = float(np.logical_or(target_mask, canvas_mask).sum())
        if union <= 0:
            return 0.0
        return intersection / union

    def _observation_payload(self) -> Dict[str, Any]:
        pen_map = np.zeros_like(self.canvas)
        pen_map[0, self.pen_y, self.pen_x] = 1.0
        obs = np.concatenate((self.canvas, self.target, pen_map), axis=0).astype(np.float32)
        return {
            "obs": obs,
            "agents": [agent.info() for agent in self.agents],
            "done": bool(self.done),
            "winner": self.winner,
            "steps": int(self.steps),
            "similarity": float(self.similarity),
            "pen_down": bool(self.pen_down),
            "brush": int(self.brush),
        }


class ShapeDrawVectorizedEnv(SimVectorEnv):
    """Runs multiple independent ShapeDrawEnv copies in lockstep."""

    def __init__(self, config: ShapeDrawConfig, num_envs: int | None = None) -> None:
        self.config = config
        resolved_envs = num_envs or getattr(config, "num_envs", 1)
        super().__init__(resolved_envs)

    def _create_env(self, index: int) -> SimEnv:
        env_config = deepcopy(self.config)
        if env_config.seed is not None:
            env_config.seed = int(env_config.seed) + index
        return ShapeDrawEnv(env_config)

    def _stack_rewards(self, reward_dicts: List[Dict[int, float]]) -> np.ndarray:
        reward_array = np.zeros((self.num_envs, self.config.num_agents), dtype=np.float32)
        for env_idx, rewards in enumerate(reward_dicts):
            for agent_id in range(self.config.num_agents):
                reward_array[env_idx, agent_id] = float(rewards.get(agent_id, 0.0))
        return reward_array

    def _stack_observations(self, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        obs_tensor = np.stack([obs["obs"] for obs in observations], axis=0)
        steps = np.asarray([obs.get("steps", 0) for obs in observations], dtype=np.int32)
        done_flags = np.asarray([obs.get("done", False) for obs in observations], dtype=np.bool_)
        similarities = np.asarray(
            [obs.get("similarity", 0.0) for obs in observations],
            dtype=np.float32,
        )
        brush = np.asarray([obs.get("brush", 0) for obs in observations], dtype=np.int32)
        pen_down = np.asarray([obs.get("pen_down", False) for obs in observations], dtype=np.bool_)
        return {
            "obs": obs_tensor,
            "agents": [obs.get("agents", []) for obs in observations],
            "done": done_flags,
            "winner": [obs.get("winner") for obs in observations],
            "steps": steps,
            "similarity": similarities,
            "brush": brush,
            "pen_down": pen_down,
        }

    def assign_agents(self, agents: list[ShapeDrawAgent]) -> None:
        if len(agents) != 1:
            raise ValueError("ShapeDraw supports exactly 1 agent")
        self.agents = agents
        template = agents[0]
        for env in self.envs:
            action_space = getattr(template, "action_space", None)
            if not isinstance(action_space, np.ndarray):
                action_count = int(getattr(env.action_space, "n", 7))
                action_space = np.arange(action_count, dtype=np.int64)
            env_agent = ShapeDrawAgent(
                agent_id=0,
                action_space=action_space,
                policy=template.policy,
                name=template.name,
            )
            env.assign_agents([env_agent])
