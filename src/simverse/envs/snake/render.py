from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

if __package__ is None or __package__.startswith("__main__"):
    _src = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_src))

import pygame
import torch

from simverse.envs.snake.config import SnakeConfig
from simverse.envs.snake.torch_env import SnakeTorchEnv
from simverse.policies.simple import SimplePolicy


def _load_policy_from_checkpoint(
    checkpoint_path: str,
    obs_space,
    action_space,
    device: str,
) -> torch.nn.Module:
    state = pickle.loads(Path(checkpoint_path).read_bytes())
    agents_state = state.get("agents", [])
    if not agents_state:
        raise ValueError(f"No agent policy found in checkpoint: {checkpoint_path}")

    policy = SimplePolicy(obs_space=obs_space, action_space=action_space)
    checkpoint_state_dict = agents_state[0]["policy_state_dict"]
    try:
        policy.load_state_dict(checkpoint_state_dict)
    except RuntimeError:
        normalized_state_dict = {
            (key[len("_orig_mod.") :] if key.startswith("_orig_mod.") else key): value
            for key, value in checkpoint_state_dict.items()
        }
        try:
            policy.load_state_dict(normalized_state_dict)
        except RuntimeError as exc:
            raise ValueError(
                "Checkpoint is incompatible with Snake policy architecture. "
                "Pass a checkpoint produced by `simverse.envs.snake.train`."
            ) from exc
    policy.to(device=device, dtype=torch.float32)
    policy.eval()
    return policy


def _policy_action(policy: torch.nn.Module, obs_tensor: torch.Tensor, device: str) -> int:
    actor_obs = obs_tensor.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        logits, _ = policy(actor_obs)
        action = torch.distributions.Categorical(logits=logits.float()).sample().item()
    return int(action)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Snake environment")
    parser.add_argument("--width", type=int, default=15)
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--init-length", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--cell-size", type=int, default=30)
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--mode", choices=["manual", "random", "policy"], default="manual")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--auto-reset", choices=["on", "off"], default="on")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def render(
    width: int = 15,
    height: int = 15,
    max_steps: int = 500,
    init_length: int = 3,
    episodes: int = 3,
    cell_size: int = 30,
    fps: int = 18,
    mode: str = "manual",
    checkpoint: str | None = None,
    auto_reset: bool = True,
    seed: int | None = None,
) -> None:
    if episodes <= 0:
        return

    if mode == "policy" and not checkpoint:
        raise ValueError("Policy mode requires --checkpoint")

    if seed is not None:
        torch.manual_seed(int(seed))

    config = SnakeConfig(
        width=max(5, int(width)),
        height=max(5, int(height)),
        num_agents=1,
        num_envs=1,
        max_steps=max(1, int(max_steps)),
        init_length=max(2, int(init_length)),
        seed=seed,
        policies=[],
    )
    env = SnakeTorchEnv(config=config, num_envs=1, device="cpu", dtype=torch.float32)

    policy = None
    if checkpoint:
        policy = _load_policy_from_checkpoint(
            checkpoint_path=checkpoint,
            obs_space=env.observation_space,
            action_space=env.action_space,
            device="cpu",
        )

    pygame.init()
    screen_width = env.width * cell_size
    hud_height = 54
    screen_height = env.height * cell_size + hud_height
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Simverse Snake")
    font = pygame.font.SysFont("Verdana", 18)
    clock = pygame.time.Clock()

    colors = {
        "bg": (20, 22, 27),
        "floor": (242, 245, 247),
        "wall": (52, 61, 74),
        "food": (210, 52, 62),
        "head": (40, 147, 66),
        "body": (76, 196, 112),
        "text": (240, 243, 248),
    }

    obs = env.reset()
    episode_done = False
    completed_episodes = 0

    running = True
    while running and completed_episodes < episodes:
        reset_requested = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    reset_requested = True

        if not episode_done:
            action = -1
            if mode == "random":
                action = int(torch.randint(0, env.action_space.n, (1,)).item())
            elif mode == "policy":
                if policy is None:
                    raise RuntimeError("Policy mode selected without loaded policy")
                action = _policy_action(policy, obs["obs"], device="cpu")
            else:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]:
                    action = env.ACTION_UP
                elif keys[pygame.K_DOWN]:
                    action = env.ACTION_DOWN
                elif keys[pygame.K_LEFT]:
                    action = env.ACTION_LEFT
                elif keys[pygame.K_RIGHT]:
                    action = env.ACTION_RIGHT

            step_actions = torch.as_tensor([[action]], dtype=torch.int64)
            obs, _rewards, done, _info = env.step(step_actions)
            if bool(done[0].item()):
                episode_done = True
                completed_episodes += 1
                print(
                    f"episode={completed_episodes} steps={int(env.steps[0].item())} "
                    f"score={int(env.score[0].item())} winner={int(env.winner[0].item())}"
                )

        if reset_requested and completed_episodes < episodes:
            obs = env.reset()
            episode_done = False

        if episode_done and auto_reset and completed_episodes < episodes:
            obs = env.reset()
            episode_done = False

        screen.fill(colors["bg"])

        for y in range(env.height):
            for x in range(env.width):
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                if x == 0 or y == 0 or x == env.width - 1 or y == env.height - 1:
                    pygame.draw.rect(screen, colors["wall"], rect)
                else:
                    pygame.draw.rect(screen, colors["floor"], rect)

        food_x = int(env.food_pos[0, 0].item())
        food_y = int(env.food_pos[0, 1].item())
        food_rect = pygame.Rect(food_x * cell_size, food_y * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, colors["food"], food_rect)

        length = int(env.snake_length[0].item())
        segments = env.snake_segments[0, :length, :].detach().cpu().numpy()
        for idx, segment in enumerate(segments):
            sx = int(segment[0])
            sy = int(segment[1])
            rect = pygame.Rect(
                sx * cell_size + 2,
                sy * cell_size + 2,
                cell_size - 4,
                cell_size - 4,
            )
            color = colors["head"] if idx == 0 else colors["body"]
            pygame.draw.rect(screen, color, rect, border_radius=max(2, cell_size // 7))

        status = "done" if episode_done else "running"
        hud_text = (
            f"episode {completed_episodes}/{episodes}  score={int(env.score[0].item())}  "
            f"steps={int(env.steps[0].item())}  state={status}  "
            "arrows=move  r=reset"
        )
        text = font.render(hud_text, True, colors["text"])
        screen.blit(text, (8, env.height * cell_size + 16))

        pygame.display.flip()
        clock.tick(max(1, int(fps)))

    pygame.quit()


if __name__ == "__main__":
    cli_args = parse_args()
    render(
        width=cli_args.width,
        height=cli_args.height,
        max_steps=cli_args.max_steps,
        init_length=cli_args.init_length,
        episodes=cli_args.episodes,
        cell_size=cli_args.cell_size,
        fps=cli_args.fps,
        mode=cli_args.mode,
        checkpoint=cli_args.checkpoint,
        auto_reset=cli_args.auto_reset == "on",
        seed=cli_args.seed,
    )
