"""Simverse RL library."""

from importlib import import_module
from typing import Dict

__all__: list[str] = []

_OPTIONAL_EXPORTS: Dict[str, str] = {
    "CartPoleEnv": "simverse.envs.cartpole",
    "PettingZooTennisEnv": "simverse.envs.tennis.env",
    "ShapeDrawEnv": "simverse.envs.shape_draw.env",
    "ShapeDrawVectorizedEnv": "simverse.envs.shape_draw.env",
    "ShapeDrawTorchEnv": "simverse.envs.shape_draw.torch_env",
    "BattleGridConfig": "simverse.envs.battle_grid.env",
    "BattleGridTorchEnv": "simverse.envs.battle_grid.env",
    "MazeRaceTorchEnv": "simverse.envs.maze_race.env",
    "GymTorchConfig": "simverse.envs.gym_env.env",
    "GymTorchEnv": "simverse.envs.gym_env.env",
    "SnakeAgent": "simverse.envs.snake.agent",
    "SnakeConfig": "simverse.envs.snake.config",
    "SnakeTorchEnv": "simverse.envs.snake.env",
    "RandomPolicy": "simverse.policies.random_policy",
    "quicktrain": "simverse.recipes.quickstart",
}

for name, module_path in _OPTIONAL_EXPORTS.items():
    try:
        module = import_module(module_path)
        globals()[name] = getattr(module, name)
        __all__.append(name)
    except (ModuleNotFoundError, AttributeError, ImportError):
        continue
