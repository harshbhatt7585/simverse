"""Command line helpers for Simverse."""

from __future__ import annotations

import typer

from simverse.recipes import quicktrain

app = typer.Typer(help="Simverse RL quickstart utilities")


@app.command()
def rollout(
    env_id: str = typer.Option("CartPole-v1", help="Gymnasium environment id"),
    episodes: int = typer.Option(5, help="Number of episodes to simulate"),
    max_steps: int = typer.Option(200, help="Max steps per episode"),
    render: bool = typer.Option(False, help="Enable human rendering"),
):
    """Runs the quicktrain recipe and prints summary stats."""

    stats = quicktrain(env_id=env_id, episodes=episodes, max_steps=max_steps, render=render)
    typer.echo(
        f"{stats['env_id']} | completed {stats['episodes']} episodes | "
        f"mean reward: {stats['mean_reward']:.2f} | "
        f"max reward: {stats['max_reward']:.2f}"
    )


if __name__ == "__main__":
    app()
