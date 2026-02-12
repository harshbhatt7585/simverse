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

from simverse.envs.battle_grid.config import BattleGridConfig
from simverse.envs.battle_grid.torch_env import BattleGridTorchEnv
from simverse.policies.simple import SimplePolicy


def _load_policies_from_checkpoint(
    checkpoint_path: str,
    obs_space,
    action_space,
    device: str,
    num_agents: int,
) -> list[torch.nn.Module]:
    state = pickle.loads(Path(checkpoint_path).read_bytes())
    agents_state = state.get("agents", [])
    if not agents_state:
        raise ValueError(f"No agent policy found in checkpoint: {checkpoint_path}")

    loaded: list[torch.nn.Module] = []
    for agent_idx in range(num_agents):
        src_idx = agent_idx if agent_idx < len(agents_state) else 0
        checkpoint_state_dict = agents_state[src_idx]["policy_state_dict"]
        policy = SimplePolicy(obs_space=obs_space, action_space=action_space)
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
                    "Checkpoint is incompatible with Battle Grid policy architecture. "
                    "Pass a checkpoint produced by `simverse.envs.battle_grid.train`."
                ) from exc
        policy.to(device=device, dtype=torch.float32)
        policy.eval()
        loaded.append(policy)
    return loaded


def _policy_actions(
    policies: list[torch.nn.Module], obs_tensor: torch.Tensor, device: str
) -> list[int]:
    actor_obs = obs_tensor.to(device=device, dtype=torch.float32)
    actions: list[int] = []
    with torch.no_grad():
        for policy in policies:
            logits, _ = policy(actor_obs)
            action = torch.distributions.Categorical(logits=logits.float()).sample().item()
            actions.append(int(action))
    return actions


def _manual_actions(keys, env: BattleGridTorchEnv) -> list[int]:
    # Agent 0: arrows + right ctrl/return to attack.
    action0 = env.ACTION_STAY
    if keys[pygame.K_UP]:
        action0 = env.ACTION_UP
    elif keys[pygame.K_DOWN]:
        action0 = env.ACTION_DOWN
    elif keys[pygame.K_LEFT]:
        action0 = env.ACTION_LEFT
    elif keys[pygame.K_RIGHT]:
        action0 = env.ACTION_RIGHT
    elif keys[pygame.K_RCTRL] or keys[pygame.K_RETURN]:
        action0 = env.ACTION_ATTACK

    # Agent 1: WASD + F to attack.
    action1 = env.ACTION_STAY
    if keys[pygame.K_w]:
        action1 = env.ACTION_UP
    elif keys[pygame.K_s]:
        action1 = env.ACTION_DOWN
    elif keys[pygame.K_a]:
        action1 = env.ACTION_LEFT
    elif keys[pygame.K_d]:
        action1 = env.ACTION_RIGHT
    elif keys[pygame.K_f]:
        action1 = env.ACTION_ATTACK

    return [action0, action1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Battle Grid environment")
    parser.add_argument("--width", type=int, default=13)
    parser.add_argument("--height", type=int, default=13)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-health", type=int, default=3)
    parser.add_argument("--attack-range", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--cell-size", type=int, default=36)
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--mode", choices=["manual", "random", "policy"], default="manual")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--auto-reset", choices=["on", "off"], default="on")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def render(
    width: int = 13,
    height: int = 13,
    max_steps: int = 200,
    max_health: int = 3,
    attack_range: int = 1,
    episodes: int = 5,
    cell_size: int = 36,
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

    config = BattleGridConfig(
        width=max(3, int(width)),
        height=max(3, int(height)),
        num_agents=2,
        num_envs=1,
        max_steps=max(1, int(max_steps)),
        max_health=max(1, int(max_health)),
        attack_range=max(1, int(attack_range)),
        policies=[],
    )
    env = BattleGridTorchEnv(config=config, num_envs=1, device="cpu", dtype=torch.float32)

    policies: list[torch.nn.Module] | None = None
    if checkpoint:
        policies = _load_policies_from_checkpoint(
            checkpoint_path=checkpoint,
            obs_space=env.observation_space,
            action_space=env.action_space,
            device="cpu",
            num_agents=env.num_agents,
        )

    pygame.init()
    hud_height = max(74, cell_size * 2)
    screen_width = env.width * cell_size
    screen_height = env.height * cell_size + hud_height
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Simverse Battle Grid")
    font = pygame.font.SysFont("Verdana", max(14, cell_size // 3))
    small_font = pygame.font.SysFont("Verdana", max(12, cell_size // 4))
    clock = pygame.time.Clock()

    colors = {
        "bg": (16, 19, 24),
        "grid": (238, 241, 247),
        "grid_line": (206, 214, 225),
        "agent0": (46, 115, 225),
        "agent1": (225, 104, 47),
        "dead": (88, 96, 108),
        "text": (238, 242, 248),
        "hud": (25, 30, 38),
        "winner": (110, 236, 146),
    }

    obs = env.reset()
    _ = obs
    done = False
    completed_episodes = 0
    last_reward = torch.zeros((1, env.num_agents), dtype=torch.float32)

    running = True
    while running and completed_episodes < episodes:
        reset_requested = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    reset_requested = True

        if not done:
            if mode == "random":
                actions = [
                    int(torch.randint(0, env.action_space.n, (1,)).item())
                    for _ in range(env.num_agents)
                ]
            elif mode == "policy":
                if policies is None:
                    raise RuntimeError("Policy mode selected without loaded policy")
                actions = _policy_actions(policies, obs["obs"], device="cpu")
            else:
                keys = pygame.key.get_pressed()
                actions = _manual_actions(keys, env)

            step_actions = torch.as_tensor([actions], dtype=torch.int64)
            obs, reward, done_tensor, _info = env.step(step_actions)
            last_reward = reward.detach().cpu()
            done = bool(done_tensor[0].item())

            if done:
                completed_episodes += 1
                winner = int(env.winner[0].item())
                hp0_final = int(env.health[0, 0].item())
                hp1_final = int(env.health[0, 1].item())
                print(
                    f"episode={completed_episodes} steps={int(env.steps[0].item())} "
                    f"winner={winner} hp0={hp0_final} hp1={hp1_final}"
                )

        if reset_requested and completed_episodes < episodes:
            obs = env.reset()
            done = False
            last_reward.zero_()

        if done and auto_reset and completed_episodes < episodes:
            obs = env.reset()
            done = False
            last_reward.zero_()

        screen.fill(colors["bg"])

        for gy in range(env.height):
            for gx in range(env.width):
                rect = pygame.Rect(gx * cell_size, gy * cell_size, cell_size, cell_size)
                pygame.draw.rect(screen, colors["grid"], rect)
                pygame.draw.rect(screen, colors["grid_line"], rect, width=1)

        for agent_id in range(env.num_agents):
            x = int(env.agent_pos[0, agent_id, 0].item())
            y = int(env.agent_pos[0, agent_id, 1].item())
            hp = int(env.health[0, agent_id].item())
            alive = hp > 0
            color_key = f"agent{agent_id}" if alive else "dead"
            cx = x * cell_size + cell_size // 2
            cy = y * cell_size + cell_size // 2
            radius = max(6, cell_size // 3)
            pygame.draw.circle(screen, colors[color_key], (cx, cy), radius)
            label = small_font.render(str(agent_id), True, colors["text"])
            screen.blit(label, (cx - label.get_width() // 2, cy - label.get_height() // 2))

        hud_rect = pygame.Rect(0, env.height * cell_size, screen_width, hud_height)
        pygame.draw.rect(screen, colors["hud"], hud_rect)

        winner = int(env.winner[0].item())
        if winner == 0:
            status = "winner: agent0"
        elif winner == 1:
            status = "winner: agent1"
        elif winner == env.WINNER_DRAW:
            status = "winner: draw"
        else:
            status = "running"

        hp0 = int(env.health[0, 0].item())
        hp1 = int(env.health[0, 1].item())
        steps = int(env.steps[0].item())
        r0 = float(last_reward[0, 0].item())
        r1 = float(last_reward[0, 1].item())
        line1 = (
            f"episode {completed_episodes}/{episodes}  steps={steps}/{env.max_steps}  "
            f"hp0={hp0} hp1={hp1}  r0={r0:+.2f} r1={r1:+.2f}"
        )
        text1 = font.render(line1, True, colors["text"])
        screen.blit(text1, (10, env.height * cell_size + 10))

        status_color = colors["winner"] if winner in (0, 1, env.WINNER_DRAW) else colors["text"]
        if mode == "manual":
            controls = "A0: arrows+enter | A1: WASD+F | R=reset"
        elif mode == "random":
            controls = "mode=random | R=reset"
        else:
            controls = "mode=policy | R=reset"
        line2 = f"{status} | {controls}"
        text2 = small_font.render(line2, True, status_color)
        screen.blit(text2, (10, env.height * cell_size + 38))

        pygame.display.flip()
        clock.tick(max(1, int(fps)))

    pygame.quit()


if __name__ == "__main__":
    cli_args = parse_args()
    render(
        width=cli_args.width,
        height=cli_args.height,
        max_steps=cli_args.max_steps,
        max_health=cli_args.max_health,
        attack_range=cli_args.attack_range,
        episodes=cli_args.episodes,
        cell_size=cli_args.cell_size,
        fps=cli_args.fps,
        mode=cli_args.mode,
        checkpoint=cli_args.checkpoint,
        auto_reset=cli_args.auto_reset == "on",
        seed=cli_args.seed,
    )
