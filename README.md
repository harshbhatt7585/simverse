# Simverse

Simverse is an RL library which contain pre-built ENVS, policies, and recipes. It is designed to learn and get started with experimenting RL.

## Abstraction Classes
- `AgentSpec` — creates the agent
- `ENVSpec` — creates the env
- `PolicySpec` — creates a policy
- `Simulator` - Upper level class to run/train the simulation

## Agent
1. Agent plays/takes action in the simulation.
2. Each agent has their own policy.
3. Agent can takes action given action space.

## ENV
1. A universe where the simulation runs.
2. ENV contains multiple trainable agents and NPCs.
3. ENV has their own rules, where agent can learn follow rules to live and win.

## Policy
1. policy is an core brain of a agent.
2. An trainable agent can hold a policy, NPCs have no policy.
3. policy is the core neural network of a agent which help them to learn the ENV.
4. policy can be any neural network like Transformers, LLM, CNN, RNN, LSTM, World Models, etc.

## How to Use (development)
Create a simulator with your environment, policies, and agent count, then kick off training:

```python
from simverse.simulator import Simulator

sim = Simulator(env="farmtila", num_agents=10, policies=["simple", "transformer"])
sim.train(
    loss="ppo",
    optimizer="adam",
    epochs=100,
)
```

## Development Setup
1. Install Simverse in editable mode with the dev extras:
   ```bash
   pip install -e .[dev]
   ```
2. To use PettingZoo Atari envs (like Tennis), install:
   ```bash
   pip install -e .[pettingzoo]
   ```
   Note: the Atari dependency stack currently requires Python `<3.13`.
3. Install the Git hooks so Ruff runs automatically:
   ```bash
   pre-commit install
   ```
4. Run the hooks on demand (useful for CI or after large refactors):
   ```bash
   pre-commit run --all-files
   ```

## UV Setup (Recommended)
1. Create a local virtual environment and install dependencies:
   ```bash
   ./scripts/setup_uv.sh dev
   ```
2. If you need Tennis/PettingZoo Atari support:
   ```bash
   ./scripts/setup_uv.sh all
   ```
3. Activate environment:
   ```bash
   source .venv/bin/activate
   ```

You can also run directly with `uv`:
- `uv sync` (base)
- `uv sync --extra dev` (base + dev)
- `uv sync --extra pettingzoo` (base + PettingZoo)
- `uv sync --extra dev --extra pettingzoo` (all)

## Shape Draw Env
Train a visual agent that draws target shapes on a canvas:

```bash
python -m simverse.envs.shape_draw.train --num-envs 64 --wandb off
```

Render and control one environment on screen:
- Arrow keys move the pen
- `Space` toggles pen up/down
- `Q`/`E` decrease/increase brush size
- `R` resets

```bash
python -m simverse.envs.shape_draw.render --size 64 --scale 6 --fps 20
```

## Gym Env (Torch Fastpath)
Train a Gymnasium discrete-action env (default `CartPole-v1`) through the torch fastpath:

```bash
python -m simverse.envs.gym_env.train --env-id CartPole-v1 --num-envs 512 --episodes 120
```

Enable W&B logging:

```bash
python -m simverse.envs.gym_env.train --env-id CartPole-v1 --wandb on
```

Render random or checkpointed rollouts, with optional MP4 recording:

```bash
python -m simverse.envs.gym_env.render --env-id CartPole-v1 --episodes 3 --record on
```
