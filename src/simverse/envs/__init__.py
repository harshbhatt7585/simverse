"""Environment helpers for Simverse."""

from importlib import import_module

__all__: list[str] = []

_KNOWN_ENVS = {
    "CartPoleEnv": "simverse.envs.cartpole",
    "PettingZooTennisEnv": "simverse.envs.tennis.env",
    "ShapeDrawEnv": "simverse.envs.shape_draw.env",
    "ShapeDrawVectorizedEnv": "simverse.envs.shape_draw.env",
    "ShapeDrawTorchEnv": "simverse.envs.shape_draw.torch_env",
    "BattleGridConfig": "simverse.envs.battle_grid.env",
    "BattleGridEnv": "simverse.envs.battle_grid.env",
    "BattleGridTorchEnv": "simverse.envs.battle_grid.env",
    "MazeRaceEnv": "simverse.envs.maze_race.env",
    "MazeRaceTorchEnv": "simverse.envs.maze_race.env",
    "GymEnv": "simverse.envs.gym_env.env",
    "GymTorchConfig": "simverse.envs.gym_env.env",
    "GymTorchEnv": "simverse.envs.gym_env.env",
    "SnakeEnv": "simverse.envs.snake.env",
    "SnakeAgent": "simverse.envs.snake.agent",
    "SnakeConfig": "simverse.envs.snake.config",
    "SnakeTorchEnv": "simverse.envs.snake.env",
}

for name, module_path in _KNOWN_ENVS.items():
    try:
        module = import_module(module_path)
        globals()[name] = getattr(module, name)
        __all__.append(name)
    except (ModuleNotFoundError, AttributeError):
        continue
