# Simverse Env Package Contract

An environment package is considered complete when it includes:

- `env.py`
- `train.py`

`render.py` is optional. Use it only for environments that still need a local renderer, such as
`farmtila`. For live/replay visualization, prefer publishing frames into the shared `renderer/`
stack instead.

## Recommended Pattern

1. `env.py` holds core torch environment logic and inherits `SimEnv`.
2. `env.py` exposes:
   - a canonical `*Env` alias for the torch implementation
   - `create_env(...)` factory
3. `train.py` uses shared helpers from `simverse.abstractor.train_utils`:
   - `resolve_torch_device`
   - `resolve_rollout_dtype`
   - `configure_torch_backend`
   - `compile_policy_models`
   - `build_adam_optimizers`
   - `build_ppo_training_config`
4. If the environment needs visualization, prefer `LiveRenderServer` plus `renderer/server`
   routes over a per-env `render.py`.

Use `simverse.envs.scaffold.missing_required_files` to validate package completeness.
