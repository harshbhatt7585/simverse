"""Render helper for Simverse PettingZoo Tennis environment."""

from __future__ import annotations

import argparse
import time

from simverse.envs.tennis.env import PettingZooTennisEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render PettingZoo Tennis environment")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to render")
    parser.add_argument(
        "--max-steps", type=int, default=2_000, help="Maximum steps per episode before reset"
    )
    parser.add_argument(
        "--sleep", type=float, default=0.0, help="Optional sleep between steps in seconds"
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional reset seed")
    return parser.parse_args()


def run_render(episodes: int, max_steps: int, sleep: float, seed: int | None) -> None:
    env = PettingZooTennisEnv(render_mode="human", seed=seed)
    try:
        for episode in range(episodes):
            _obs, _infos = env.reset()
            for step in range(max_steps):
                actions = {agent: env.action_space[agent].sample() for agent in env.agents}
                _obs, _rewards, terminations, truncations, _infos = env.step(actions)
                if sleep > 0:
                    time.sleep(sleep)
                if all(terminations.values()) or all(truncations.values()):
                    print(f"Episode {episode + 1} finished after {step + 1} steps")
                    break
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    run_render(
        episodes=args.episodes,
        max_steps=args.max_steps,
        sleep=args.sleep,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
