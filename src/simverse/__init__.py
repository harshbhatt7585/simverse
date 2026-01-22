"""Simverse RL library."""

from importlib import import_module
from typing import Dict

__all__: list[str] = []

_OPTIONAL_EXPORTS: Dict[str, str] = {
    "CartPoleEnv": "simverse.envs.cartpole",
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
