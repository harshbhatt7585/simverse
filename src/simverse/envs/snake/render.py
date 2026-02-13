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
from simverse.render_cli import build_render_parser

SIDEBAR_WIDTH = 340
COLORS = {
    "bg": (16, 20, 24),
    "floor": (242, 245, 247),
    "wall": (55, 67, 82),
    "food": (210, 52, 62),
    "head": (40, 147, 66),
    "body": (76, 196, 112),
    "text": (237, 242, 247),
    "panel_bg": (23, 28, 34),
    "panel_border": (61, 74, 90),
    "panel_heading": (161, 207, 255),
    "panel_label": (148, 164, 184),
    "panel_value": (233, 240, 248),
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
    title_font: pygame.font.Font,
    cell_size: int,
    obs: np.ndarray,
    panel_title: str,
    panel_metrics: list[tuple[str, str]],
    panel_footer: list[str] | None = None,
) -> None:
    height = int(obs.shape[1])
    width = int(obs.shape[2])
    grid_w_px = width * cell_size

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

    panel_rect = pygame.Rect(grid_w_px, 0, SIDEBAR_WIDTH, height * cell_size)
    pygame.draw.rect(screen, COLORS["panel_bg"], panel_rect)
    pygame.draw.line(
        screen,
        COLORS["panel_border"],
        (grid_w_px, 0),
        (grid_w_px, height * cell_size),
        width=2,
    )

    title = title_font.render(panel_title, True, COLORS["panel_heading"])
    screen.blit(title, (grid_w_px + 16, 14))

    y = 54
    for label, value in panel_metrics:
        label_text = font.render(label, True, COLORS["panel_label"])
        value_text = font.render(value, True, COLORS["panel_value"])
        screen.blit(label_text, (grid_w_px + 16, y))
        screen.blit(value_text, (grid_w_px + 140, y))
        y += 24

    if panel_footer:
        y += 8
        for line in panel_footer:
            text = font.render(line, True, COLORS["panel_label"])
            screen.blit(text, (grid_w_px + 16, y))
            y += 22

    pygame.display.flip()


def _termination_label(reason: int) -> str:
    if reason == 1:
        return "wall"
    if reason == 2:
        return "self"
    if reason == 3:
        return "timeout"
    if reason == 4:
        return "full-grid"
    return "none"


def _extract_reward_value(rewards) -> float:
    if rewards is None:
        return 0.0
    if isinstance(rewards, dict):
        if "reward" in rewards:
            try:
                return float(rewards["reward"])
            except (TypeError, ValueError):
                return 0.0
        if not rewards:
            return 0.0
        rewards = next(iter(rewards.values()))
    if isinstance(rewards, np.ndarray):
        if rewards.size == 0:
            return 0.0
        rewards = rewards.reshape(-1)[0]
    elif isinstance(rewards, (list, tuple)):
        if not rewards:
            return 0.0
        first = rewards[0]
        if isinstance(first, dict):
            total = 0.0
            found = False
            for item in rewards:
                if not isinstance(item, dict):
                    continue
                value = item.get("reward")
                if value is None:
                    continue
                try:
                    total += float(value)
                    found = True
                except (TypeError, ValueError):
                    continue
            if found:
                return total
        rewards = first
    try:
        return float(rewards)
    except (TypeError, ValueError):
        return 0.0


def _infer_length_from_obs(obs: np.ndarray) -> int:
    if obs.ndim != 3 or obs.shape[0] < 4:
        return 0
    head_count = int(np.sum(obs[2] > 0.5))
    body_count = int(np.sum(obs[3] > 0.5))
    return head_count + body_count


def _infer_pos_from_obs(layer: np.ndarray) -> tuple[int, int] | None:
    if layer.ndim != 2:
        return None
    coords = np.argwhere(layer > 0.5)
    if coords.size == 0:
        return None
    y, x = coords[0]
    return (int(x), int(y))


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
    screen = pygame.display.set_mode((grid_w * cell_size + SIDEBAR_WIDTH, grid_h * cell_size))
    pygame.display.set_caption("Simverse Snake Replay")
    font = pygame.font.SysFont("Verdana", 18)
    title_font = pygame.font.SysFont("Verdana", 22, bold=True)
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

        replay_episode_reward = 0.0
        prev_episode: int | None = None
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
                    (grid_w * cell_size + SIDEBAR_WIDTH, grid_h * cell_size)
                )

            info = frame.get("info", {}) if isinstance(frame.get("info", {}), dict) else {}
            step = _extract_scalar_int(frame.get("step"), default=0)
            episode = _extract_scalar_int(frame.get("episode"), default=0)
            inferred_length = _infer_length_from_obs(obs)
            length = _extract_scalar_int(
                info.get("snake_length", info.get("slength")),
                default=inferred_length,
            )
            length = max(length, inferred_length)
            term_reason = _extract_scalar_int(info.get("termination_reason"), default=0)
            reward = _extract_reward_value(frame.get("rewards"))
            if prev_episode is None or episode != prev_episode:
                replay_episode_reward = 0.0
                prev_episode = episode
            replay_episode_reward += reward
            done = bool(frame.get("done", False))
            status = "done" if done else "running"
            head_pos = info.get("head_pos")
            if head_pos is None:
                head_pos = _infer_pos_from_obs(obs[2])
            food_pos = info.get("food_pos")
            if food_pos is None:
                food_pos = _infer_pos_from_obs(obs[1])

            metrics = [
                ("Replay", path.name),
                ("Episode", str(episode)),
                ("Step", str(step)),
                ("State", status),
                ("Termination", f"{_termination_label(term_reason)} ({term_reason})"),
                ("Length", str(length)),
                ("Reward", f"{reward:.3f}"),
                ("Ep Reward", f"{replay_episode_reward:.3f}"),
                ("Head", str(head_pos)),
                ("Food", str(food_pos)),
                ("FPS", str(max(1, int(fps)))),
            ]
            _draw_obs_frame(
                screen=screen,
                font=font,
                title_font=title_font,
                cell_size=cell_size,
                obs=obs,
                panel_title="Snake Replay",
                panel_metrics=metrics,
                panel_footer=["ESC to quit"],
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
    parser = build_render_parser(
        "Render Snake environment",
        [
            ("width", {"default": 15}),
            ("height", {"default": 15}),
            ("max_steps", {"default": 500}),
            ("init_length", {"default": 3}),
            ("episodes", {"default": 3}),
            ("cell_size", {"default": 30}),
            ("fps", {"default": 18}),
            "mode",
            "checkpoint",
            "auto_reset",
            "seed",
            ("replay", {"help": "Path to a replay JSON file"}),
            ("replay_dir", {"help": "Directory containing replay JSON files"}),
            ("loop", {"help": "Loop replay playback"}),
            ("watch", {"help": "Watch replay dir for new files"}),
            ("poll", {"help": "Replay dir poll interval"}),
        ],
    )
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
            obs_space=env.observation_space["obs"],
            action_space=env.action_space,
            device="cpu",
        )

    pygame.init()
    screen_width = env.width * cell_size + SIDEBAR_WIDTH
    screen_height = env.height * cell_size
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Simverse Snake")
    font = pygame.font.SysFont("Verdana", 18)
    title_font = pygame.font.SysFont("Verdana", 22, bold=True)
    clock = pygame.time.Clock()

    obs = env.reset()
    episode_done = False
    completed_episodes = 0
    episode_reward = 0.0
    last_reward = 0.0
    last_action = -1

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
            obs, rewards, done, _info = env.step(step_actions)
            last_action = int(action)
            last_reward = float(rewards[0, 0].item())
            episode_reward += last_reward
            if bool(done[0].item()):
                episode_done = True
                completed_episodes += 1
                print(
                    f"episode={completed_episodes} steps={int(env.steps[0].item())} "
                    f"score={int(env.score[0].item())} "
                    f"term={int(env.termination_reason[0].item())}"
                )

        if reset_requested and completed_episodes < episodes:
            obs = env.reset()
            episode_done = False
            episode_reward = 0.0
            last_reward = 0.0
            last_action = -1

        if episode_done and auto_reset and completed_episodes < episodes:
            obs = env.reset()
            episode_done = False
            episode_reward = 0.0
            last_reward = 0.0
            last_action = -1
        elif episode_done and not auto_reset:
            running = False

        frame_obs = obs["obs"][0].detach().cpu().numpy()
        head_pos = (
            int(env.snake_segments[0, 0, 0].item()),
            int(env.snake_segments[0, 0, 1].item()),
        )
        food_pos = (
            int(env.food_pos[0, 0].item()),
            int(env.food_pos[0, 1].item()),
        )
        length = int(env.snake_length[0].item())
        term_reason = int(env.termination_reason[0].item())
        status = "done" if episode_done else "running"
        metrics = [
            ("Episode", f"{completed_episodes}/{episodes}"),
            ("Mode", mode),
            ("State", status),
            ("Termination", f"{_termination_label(term_reason)} ({term_reason})"),
            ("Step", f"{int(env.steps[0].item())}/{max_steps}"),
            ("Score", str(int(env.score[0].item()))),
            ("Length", str(length)),
            ("Last Reward", f"{last_reward:.3f}"),
            ("Ep Reward", f"{episode_reward:.3f}"),
            ("Action", str(last_action)),
            ("Head", str(head_pos)),
            ("Food", str(food_pos)),
            ("FPS", str(max(1, int(fps)))),
        ]
        _draw_obs_frame(
            screen=screen,
            font=font,
            title_font=title_font,
            cell_size=cell_size,
            obs=frame_obs,
            panel_title="Snake Metrics",
            panel_metrics=metrics,
            panel_footer=["Arrows to move", "R to reset", "Close window to exit"],
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
