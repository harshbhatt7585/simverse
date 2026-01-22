"""Environment helpers for Simverse."""

from importlib import import_module

__all__: list[str] = []

_KNOWN_ENVS = {
    "CartPoleEnv": "simverse.envs.cartpole",
}

for name, module_path in _KNOWN_ENVS.items():
    try:
        module = import_module(module_path)
        globals()[name] = getattr(module, name)
        __all__.append(name)
    except (ModuleNotFoundError, AttributeError):
        continue
