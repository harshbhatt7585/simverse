from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from typing import Any

_ARG_TEMPLATES: dict[str, tuple[tuple[str, ...], dict[str, Any]]] = {
    "width": (("--width",), {"type": int, "default": 15, "help": "Grid width"}),
    "height": (("--height",), {"type": int, "default": 15, "help": "Grid height"}),
    "size": (
        ("--size",),
        {"type": int, "default": 15, "help": "Square grid width/height"},
    ),
    "num_agents": (
        ("--num-agents",),
        {"type": int, "default": 2, "help": "Number of agents"},
    ),
    "episodes": (("--episodes",), {"type": int, "default": 3, "help": "Episode count"}),
    "max_steps": (("--max-steps",), {"type": int, "default": 500, "help": "Max steps"}),
    "max_frames": (("--max-frames",), {"type": int, "default": 0}),
    "max_episodes": (("--max-episodes",), {"type": int, "default": 0}),
    "init_length": (("--init-length",), {"type": int, "default": 3}),
    "max_health": (("--max-health",), {"type": int, "default": 3}),
    "attack_range": (("--attack-range",), {"type": int, "default": 1}),
    "cell_size": (
        ("--cell-size",),
        {"type": int, "default": 30, "help": "Cell size in pixels"},
    ),
    "cell": (("--cell",), {"type": int, "default": 36, "help": "Cell size in pixels"}),
    "fps": (("--fps",), {"type": int, "default": 20, "help": "Render FPS"}),
    "mode": (
        ("--mode",),
        {"choices": ["manual", "random", "policy"], "default": "manual"},
    ),
    "checkpoint": (("--checkpoint",), {"type": str, "default": None}),
    "auto_reset": (
        ("--auto-reset",),
        {"choices": ["on", "off"], "default": "on"},
    ),
    "seed": (("--seed",), {"type": int, "default": None}),
    "replay": (("--replay",), {"type": str, "default": None}),
    "replay_dir": (("--replay-dir",), {"type": str, "default": None}),
    "loop": (("--loop",), {"action": "store_true"}),
    "watch": (("--watch",), {"action": "store_true"}),
    "poll": (("--poll",), {"type": float, "default": 1.0}),
    "manual_flag": (
        ("--manual",),
        {
            "dest": "auto",
            "action": "store_false",
            "help": "Disable auto-run and use keyboard controls",
        },
    ),
    "no_auto_reset_flag": (
        ("--no-auto-reset",),
        {"dest": "auto_reset", "action": "store_false", "help": "Stop after an episode ends"},
    ),
    "env_id": (("--env-id",), {"type": str, "default": "CartPole-v1"}),
    "record": (("--record",), {"choices": ["on", "off"], "default": "off"}),
    "record_dir": (("--record-dir",), {"type": str, "default": "recordings/gym_env/videos"}),
}


def build_render_parser(
    description: str,
    options: Iterable[str | tuple[str, Mapping[str, Any]]],
    *,
    defaults: Mapping[str, Any] | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    for entry in options:
        key, overrides = (entry, {}) if isinstance(entry, str) else (entry[0], dict(entry[1]))
        if key not in _ARG_TEMPLATES:
            raise KeyError(f"Unknown render CLI option: {key}")
        flags, kwargs = _ARG_TEMPLATES[key]
        merged = dict(kwargs)
        merged.update(overrides)
        parser.add_argument(*flags, **merged)
    if defaults:
        parser.set_defaults(**defaults)
    return parser
