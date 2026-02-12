from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

if __package__ is None or __package__.startswith("__main__"):
    _src = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_src))

import numpy as np
import pygame
import torch

from simverse.envs.snake.config import SnakeConfig
from simverse.envs.snake.torch_env import SnakeTorchEnv
from simverse.policies.simple import SimplePolicy

HUD_HEIGHT = 54
COLORS = {
    "bg": (20, 22, 27),
    "floor": (242, 245, 247),
    "wall": (52, 61, 74),
    "food": (210, 52, 62),
    "head": (40, 147, 66),
    "body": (76, 196, 112),
    "text": (240, 243, 248),
}


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


def _extract_scalar_int(value, default: int = 0) -> int:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.reshape(-1)[0]
    elif isinstance(value, (list, tuple)):
        if not value:
            return default
        value = value[0]
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _draw_obs_frame(
    *,
    screen: pygame.Surface,
    font: pygame.font.Font,
    cell_size: int,
    obs: np.ndarray,
    hud_text: str,
) -> None:
    height = int(obs.shape[1])
    width = int(obs.shape[2])

    walls = obs[0]
    food = obs[1] if obs.shape[0] > 1 else np.zeros_like(walls)
    head = obs[2] if obs.shape[0] > 2 else np.zeros_like(walls)
    body = obs[3] if obs.shape[0] > 3 else np.zeros_like(walls)

    screen.fill(COLORS["bg"])

    for y in range(height):
        for x in range(width):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            if walls[y, x] > 0.5:
                pygame.draw.rect(screen, COLORS["wall"], rect)
            else:
                pygame.draw.rect(screen, COLORS["floor"], rect)

    food_cells = np.argwhere(food > 0.5)
    for fy, fx in food_cells:
        rect = pygame.Rect(int(fx) * cell_size, int(fy) * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, COLORS["food"], rect)

    body_cells = np.argwhere(body > 0.5)
    for by, bx in body_cells:
        rect = pygame.Rect(
            int(bx) * cell_size + 2,
            int(by) * cell_size + 2,
            cell_size - 4,
            cell_size - 4,
        )
        pygame.draw.rect(screen, COLORS["body"], rect, border_radius=max(2, cell_size // 7))

    head_cells = np.argwhere(head > 0.5)
    for hy, hx in head_cells:
        rect = pygame.Rect(
            int(hx) * cell_size + 2,
            int(hy) * cell_size + 2,
            cell_size - 4,
            cell_size - 4,
        )
        pygame.draw.rect(screen, COLORS["head"], rect, border_radius=max(2, cell_size // 7))

    text = font.render(hud_text, True, COLORS["text"])
    screen.blit(text, (8, height * cell_size + 16))
    pygame.display.flip()


def _render_replay(
    *,
    replay: str | None,
    replay_dir: str | None,
    cell_size: int,
    fps: int,
    loop: bool,
    watch: bool,
    poll: float,
    width: int,
    height: int,
) -> None:
    replay_paths: list[Path] = []
    if replay:
        replay_path = Path(replay)
        if not replay_path.exists():
            raise SystemExit(f"Replay file not found: {replay_path}")
        replay_paths = [replay_path]
    else:
        replay_dir_path = Path(replay_dir or "")
        if not replay_dir_path.exists():
            raise SystemExit(f"Replay directory not found: {replay_dir_path}")
        replay_paths = sorted(replay_dir_path.glob("*.json"))
        if not replay_paths and not watch:
            raise SystemExit(f"No replay JSON files found in {replay_dir_path}")

    pygame.init()
    grid_w = max(5, int(width))
    grid_h = max(5, int(height))
    screen = pygame.display.set_mode((grid_w * cell_size, grid_h * cell_size + HUD_HEIGHT))
    pygame.display.set_caption("Simverse Snake Replay")
    font = pygame.font.SysFont("Verdana", 18)
    clock = pygame.time.Clock()

    seen: set[Path] = set()

    def _handle_events() -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def _play_single_replay(path: Path) -> bool:
        nonlocal screen, grid_w, grid_h
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            return True

        frames = data.get("frames", [])
        if not isinstance(frames, list) or not frames:
            return True

        for frame in frames:
            if not _handle_events():
                return False

            obs = np.asarray(frame.get("observation"), dtype=np.float32)
            if obs.ndim != 3 or obs.shape[0] < 4:
                continue

            frame_h = int(obs.shape[1])
            frame_w = int(obs.shape[2])
            if (frame_w, frame_h) != (grid_w, grid_h):
                grid_w, grid_h = frame_w, frame_h
                screen = pygame.display.set_mode(
                    (grid_w * cell_size, grid_h * cell_size + HUD_HEIGHT)
                )

            info = frame.get("info", {}) if isinstance(frame.get("info", {}), dict) else {}
            step = _extract_scalar_int(frame.get("step"), default=0)
            score = _extract_scalar_int(info.get("score"), default=0)
            winner = _extract_scalar_int(info.get("winner"), default=-1)
            done = bool(frame.get("done", False))
            status = "done" if done else "running"

            hud_text = (
                f"replay={path.name} step={step} score={score} " f"winner={winner} state={status}"
            )
            _draw_obs_frame(
                screen=screen,
                font=font,
                cell_size=cell_size,
                obs=obs,
                hud_text=hud_text,
            )
            clock.tick(max(1, int(fps)))
        return True

    try:
        if replay:
            while True:
                if not _play_single_replay(replay_paths[0]):
                    break
                if not loop:
                    break
        elif watch:
            replay_dir_path = Path(replay_dir or "")
            while True:
                if not _handle_events():
                    break
                files = sorted(replay_dir_path.glob("*.json"))
                new_files = [path for path in files if path not in seen]
                if not new_files:
                    time.sleep(max(float(poll), 0.1))
                    continue
                for path in new_files:
                    if not _play_single_replay(path):
                        return
                    seen.add(path)
        else:
            while True:
                for path in replay_paths:
                    if not _play_single_replay(path):
                        return
                if not loop:
                    break
    finally:
        pygame.quit()


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
    parser.add_argument("--replay", type=str, default=None, help="Path to a replay JSON file")
    parser.add_argument(
        "--replay-dir",
        type=str,
        default=None,
        help="Directory containing replay JSON files",
    )
    parser.add_argument("--loop", action="store_true", help="Loop replay playback")
    parser.add_argument("--watch", action="store_true", help="Watch replay dir for new files")
    parser.add_argument("--poll", type=float, default=1.0, help="Replay dir poll interval")
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
    replay: str | None = None,
    replay_dir: str | None = None,
    loop: bool = False,
    watch: bool = False,
    poll: float = 1.0,
) -> None:
    if replay or replay_dir:
        _render_replay(
            replay=replay,
            replay_dir=replay_dir,
            cell_size=cell_size,
            fps=fps,
            loop=loop,
            watch=watch,
            poll=poll,
            width=width,
            height=height,
        )
        return

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
    screen_height = env.height * cell_size + HUD_HEIGHT
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Simverse Snake")
    font = pygame.font.SysFont("Verdana", 18)
    clock = pygame.time.Clock()

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
        elif episode_done and not auto_reset:
            running = False

        frame_obs = obs["obs"][0].detach().cpu().numpy()
        status = "done" if episode_done else "running"
        hud_text = (
            f"episode {completed_episodes}/{episodes}  score={int(env.score[0].item())}  "
            f"steps={int(env.steps[0].item())}  state={status}  arrows=move  r=reset"
        )
        _draw_obs_frame(
            screen=screen,
            font=font,
            cell_size=cell_size,
            obs=frame_obs,
            hud_text=hud_text,
        )

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
        replay=cli_args.replay,
        replay_dir=cli_args.replay_dir,
        loop=cli_args.loop,
        watch=cli_args.watch,
        poll=cli_args.poll,
    )
