# Simverse Env Package Contract

An environment package is considered complete when it includes:

- `env.py`
- `train.py`
- `render.py`

## Recommended Pattern

1. `torch_env.py` holds core environment logic and inherits `SimTorchEnv`.
2. `env.py` exposes:
   - a canonical `*Env` class that extends `*TorchEnv`
   - `create_env(...)` factory
3. `train.py` uses shared helpers from `simverse.abstractor.train_utils`:
   - `resolve_torch_device`
   - `resolve_rollout_dtype`
   - `configure_torch_backend`
   - `compile_policy_models`
   - `build_adam_optimizers`
   - `build_ppo_training_config`

Use `simverse.envs.scaffold.missing_required_files` to validate package completeness.
