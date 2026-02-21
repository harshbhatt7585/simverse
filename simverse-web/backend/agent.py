from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@dataclass
class AgentTurn:
    status: str
    reply: str


example_env_code = """
from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
from simverse.abstractor.simenv import SimEnv
from simverse.envs.farmtila.agent import FarmtilaAgent
from simverse.envs.farmtila.config import FarmtilaConfig


class FarmtilaTorchEnv(SimEnv):
    HARVEST_ACTION = 4
    PICKUP_ACTION = 5
    ACTION_SPACE = gym.spaces.Discrete(6)
    LAND_EMPTY = 0
    LAND_OWNED = 1

    def __init__(
        self,
        config: FarmtilaConfig,
        num_envs: int | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.config = config
        if self.config.num_agents != 2:
            raise ValueError("Competitive Farmtila requires exactly 2 agents")

        self.num_envs = self._resolve_num_envs(num_envs, config)
        self.width = config.width
        self.height = config.height
        self.num_agents = config.num_agents
        self.agents: list[FarmtilaAgent] = []

        self.register_buffer(
            "seed_grid",
            torch.zeros(self.num_envs, self.width, self.height, dtype=torch.int64),
        )
        self.register_buffer(
            "owner_grid",
            torch.full(
                (self.num_envs, self.width, self.height),
                -1,
                dtype=torch.int64,
            ),
        )
        self.register_buffer(
            "farm_grid",
            torch.zeros(self.num_envs, self.width, self.height, dtype=torch.int64),
        )
        self.register_buffer(
            "agent_pos",
            torch.zeros(self.num_envs, self.num_agents, 2, dtype=torch.int64),
        )
        self.register_buffer(
            "inventory",
            torch.zeros(self.num_envs, self.num_agents, dtype=torch.int64),
        )
        self.register_buffer(
            "harvested_tiles",
            torch.zeros(self.num_envs, self.num_agents, dtype=torch.int64),
        )
        self.register_buffer(
            "seeds_spawned",
            torch.zeros(self.num_envs, dtype=torch.int64),
        )
        self.register_buffer("steps", torch.zeros(self.num_envs, dtype=torch.int64))
        self.register_buffer("done", torch.zeros(self.num_envs, dtype=torch.bool))
        self.register_buffer(
            "winner",
            torch.full((self.num_envs,), -1, dtype=torch.int64),
        )
        self.register_buffer(
            "delta_x",
            torch.tensor([0, 0, -1, 1, 0, 0], dtype=torch.int64),
        )
        self.register_buffer(
            "delta_y",
            torch.tensor([-1, 1, 0, 0, 0, 0], dtype=torch.int64),
        )
        self.register_buffer(
            "env_idx",
            torch.arange(self.num_envs, dtype=torch.int64),
        )
        self.to(self.device)

    @property
    def action_space(self):
        return self.ACTION_SPACE

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=-1,
            high=max(self.num_agents, self.LAND_OWNED, 1),
            shape=(5, self.width, self.height),
            dtype=np.float32,
        )

    def assign_agents(self, agents: list[FarmtilaAgent]) -> None:
        self._assign_agents(agents, expected_count=2, label="Competitive Farmtila")

    def reset(self) -> Dict[str, torch.Tensor]:
        self.seed_grid.zero_()
        self.owner_grid.fill_(-1)
        self.farm_grid.zero_()
        self.inventory.zero_()
        self.harvested_tiles.zero_()
        self.seeds_spawned.zero_()
        self._reset_episode_state(winner_none=-1)
        self._spawn_agents()
        self._spawn_seeds_if_due(force=True)
        return self._get_observation()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        action_tensor = self._normalize_actions(actions)
        rewards = self._empty_rewards()
        active_mask = ~self.done
        env_idx = self.env_idx

        step_cost = float(getattr(self.config, "step_cost", 0.0))
        if step_cost != 0.0:
            rewards += (-step_cost) * active_mask.unsqueeze(1).to(self.dtype)

        prev_score_delta = (self.harvested_tiles[:, 0] - self.harvested_tiles[:, 1]).to(self.dtype)

        for agent_id in range(self.num_agents):
            action = action_tensor[:, agent_id]
            has_action = action >= 0
            active_action = has_action & active_mask
            action_index = torch.clamp(action, min=0, max=5)

            dx = self.delta_x[action_index] * active_action.to(self.delta_x.dtype)
            dy = self.delta_y[action_index] * active_action.to(self.delta_y.dtype)

            pos_x = self.agent_pos[:, agent_id, 0]
            pos_y = self.agent_pos[:, agent_id, 1]
            new_x = torch.clamp(pos_x + dx, 0, self.width - 1)
            new_y = torch.clamp(pos_y + dy, 0, self.height - 1)

            pos_x = torch.where(active_action, new_x, pos_x)
            pos_y = torch.where(active_action, new_y, pos_y)
            self.agent_pos[:, agent_id, 0] = pos_x
            self.agent_pos[:, agent_id, 1] = pos_y

            pickup = active_action & (self.seed_grid[env_idx, pos_x, pos_y] > 0)
            pickup_idx = env_idx[pickup]
            self.seed_grid[pickup_idx, pos_x[pickup], pos_y[pickup]] = 0
            self.inventory[pickup, agent_id] += 1

            harvest_action = (action == self.HARVEST_ACTION) & active_action
            can_spend = harvest_action & (self.inventory[:, agent_id] > 0)
            owner = self.owner_grid[env_idx, pos_x, pos_y]
            target_is_other = can_spend & (owner != agent_id)
            target_idx = env_idx[target_is_other]
            if target_idx.numel() > 0:
                prev_owner = owner[target_is_other]
                self.inventory[target_is_other, agent_id] -= 1
                self.owner_grid[target_idx, pos_x[target_is_other], pos_y[target_is_other]] = (
                    agent_id
                )
                self.farm_grid[target_idx, pos_x[target_is_other], pos_y[target_is_other]] = (
                    self.LAND_OWNED
                )
                self.harvested_tiles[target_is_other, agent_id] += 1

                had_prev_owner = prev_owner >= 0
                if torch.any(had_prev_owner):
                    prev_owner_envs = target_idx[had_prev_owner]
                    prev_owner_ids = prev_owner[had_prev_owner]
                    self.harvested_tiles[prev_owner_envs, prev_owner_ids] = torch.clamp(
                        self.harvested_tiles[prev_owner_envs, prev_owner_ids] - 1,
                        min=0,
                    )

        self.steps[active_mask] += 1
        self._spawn_seeds_if_due()
        self._check_episode_end(rewards)

        score_delta_reward = float(getattr(self.config, "score_delta_reward", 1.0))
        if score_delta_reward != 0.0:
            score_delta = (self.harvested_tiles[:, 0] - self.harvested_tiles[:, 1]).to(self.dtype)
            delta_change = (score_delta - prev_score_delta) * score_delta_reward
            rewards[:, 0] += delta_change
            rewards[:, 1] -= delta_change

        obs = self._get_observation()
        info = self._build_info()
        return obs, rewards, self.done.clone(), info

    def _normalize_actions(self, actions: torch.Tensor | None) -> torch.Tensor:
        return self._normalize_action_matrix(actions)

    def _spawn_agents(self) -> None:
        positions = torch.stack(
            (
                torch.randint(
                    0,
                    self.width,
                    (self.num_envs, self.num_agents),
                    device=self.device,
                ),
                torch.randint(
                    0,
                    self.height,
                    (self.num_envs, self.num_agents),
                    device=self.device,
                ),
            ),
            dim=-1,
        )
        self.agent_pos.copy_(positions)

    def _spawn_seeds_if_due(self, *, force: bool = False) -> None:
        if self.config.spawn_seed_every <= 0 and not force:
            return
        due_mask = (force | ((self.steps % self.config.spawn_seed_every) == 0)) & (~self.done)
        due_env_indices = torch.nonzero(due_mask, as_tuple=True)[0]
        if due_env_indices.numel() == 0:
            return
        total_cells = self.width * self.height
        spawn_cap = min(int(self.config.seeds_per_spawn), total_cells)
        if spawn_cap <= 0:
            return
        budgets = torch.clamp(
            self.config.total_seeds_per_episode - self.seeds_spawned[due_env_indices],
            min=0,
        )
        spawn_counts = torch.clamp(budgets, max=spawn_cap)

        random_scores = torch.rand(
            (due_env_indices.shape[0], total_cells),
            device=self.device,
        )
        flat_indices = torch.topk(
            random_scores,
            k=spawn_cap,
            dim=1,
            largest=False,
        ).indices

        xs = flat_indices // self.height
        ys = flat_indices % self.height

        due_env_grid = due_env_indices.unsqueeze(1).expand(-1, spawn_cap)
        existing_seed = self.seed_grid[due_env_grid, xs, ys]
        existing_farm = self.farm_grid[due_env_grid, xs, ys]
        within_budget = torch.arange(spawn_cap, device=self.device).unsqueeze(
            0
        ) < spawn_counts.unsqueeze(1)
        place_mask = within_budget & (existing_seed == 0) & (existing_farm == 0)

        self.seed_grid[due_env_grid, xs, ys] = torch.where(
            place_mask,
            torch.ones_like(existing_seed),
            existing_seed,
        )
        self.seeds_spawned[due_env_indices] += place_mask.sum(dim=1)

    def _check_episode_end(self, rewards: torch.Tensor) -> None:
        max_steps_mask = self.steps >= self.config.max_steps

        budgets = self.config.total_seeds_per_episode - self.seeds_spawned
        no_budget = budgets <= 0
        no_seed_on_map = self.seed_grid.view(self.num_envs, -1).sum(dim=1) == 0
        no_inventory = self.inventory.sum(dim=1) == 0
        exhausted_mask = no_budget & no_seed_on_map & no_inventory

        end_mask = (~self.done) & (max_steps_mask | exhausted_mask)
        if not torch.any(end_mask):
            return

        score0 = self.harvested_tiles[:, 0]
        score1 = self.harvested_tiles[:, 1]
        winner_ids = torch.where(score0 > score1, 0, torch.where(score1 > score0, 1, -1))

        self.done |= end_mask
        self.winner = torch.where(end_mask, winner_ids, self.winner)

        terminal = float(getattr(self.config, "terminal_win_reward", 1.0))
        if terminal != 0.0:
            envs0 = self.env_idx[end_mask & (winner_ids == 0)]
            envs1 = self.env_idx[end_mask & (winner_ids == 1)]
            rewards[envs0, 0] += terminal
            rewards[envs0, 1] -= terminal
            rewards[envs1, 1] += terminal
            rewards[envs1, 0] -= terminal

    def _get_observation(self) -> Dict[str, torch.Tensor]:
        agent_grid = torch.zeros(
            (self.num_envs, self.width, self.height),
            dtype=self.dtype,
            device=self.device,
        )
        inventory_grid = torch.zeros_like(agent_grid)

        env_idx = self.env_idx
        for agent_id in range(self.num_agents):
            x = self.agent_pos[:, agent_id, 0]
            y = self.agent_pos[:, agent_id, 1]
            agent_grid[env_idx, x, y] = float(agent_id + 1)
            inventory_grid[env_idx, x, y] = self.inventory[:, agent_id].to(self.dtype)

        obs = torch.stack(
            [
                self.seed_grid.to(self.dtype),
                self.owner_grid.to(self.dtype),
                self.farm_grid.to(self.dtype),
                agent_grid,
                inventory_grid,
            ],
            dim=1,
        )

        return self._pack_observation_dict(obs)


FarmtilaEnv = FarmtilaTorchEnv


def create_env(
    config: FarmtilaConfig,
    *,
    num_envs: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> FarmtilaTorchEnv:
    return FarmtilaTorchEnv(config=config, num_envs=num_envs, device=device, dtype=dtype)
"""


example_train_code = """
from __future__ import annotations

import random

import torch
from simverse.abstractor.train_utils import run_ppo_training
from simverse.envs.farmtila.agent import FarmtilaAgent
from simverse.envs.farmtila.config import FarmtilaConfig, build_training_config
from simverse.envs.farmtila.env import FarmtilaEnv, create_env
from simverse.policies.simple import SimplePolicy


def agent_factory(agent_id: int, policy: torch.nn.Module, env: FarmtilaEnv) -> FarmtilaAgent:
    return FarmtilaAgent(
        agent_id=agent_id,
        position=(
            random.randint(0, env.config.width - 1),
            random.randint(0, env.config.height - 1),
        ),
        action_space=env.action_space,
        policy=policy,
    )


def train(use_wandb: bool = True, use_compile: bool = True):
    training_config = build_training_config(
        num_agents=2,
        num_envs=2048,
        max_steps=1500,
        episodes=100,
        training_epochs=1,
        lr=0.001,
        clip_epsilon=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        total_seeds=500,
        batch_size=None,
        buffer_size=None,
        dtype=torch.float16,
    )

    config = FarmtilaConfig(
        width=training_config["width"],
        height=training_config["height"],
        num_agents=training_config["num_agents"],
        num_envs=training_config["num_envs"],
        total_seeds_per_episode=training_config["total_seeds"],
        max_steps=training_config["max_steps"],
        spawn_seed_every=100,
        seeds_per_spawn=10,
        policies=[],
    )
    env = create_env(
        config=config,
        num_envs=training_config["num_envs"],
        device=training_config["device"],
        dtype=training_config["dtype"],
    )
    run_ppo_training(
        env=env,
        training_config=training_config,
        agent_factory=agent_factory,
        policy_factory=lambda obs_space, action_space: SimplePolicy(
            obs_space=obs_space,
            action_space=action_space,
        ),
        title="Farmtila Training",
        run_name="ppo-training",
        episode_save_dir="recordings/farmtila",
        use_wandb=use_wandb,
        use_compile=use_compile,
        policy_name_prefix="simple_agent",
    )


if __name__ == "__main__":
    train()
"""

example_config_code = """
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class FarmtilaConfig:
    width: int = 50
    height: int = 50
    num_agents: int = 2
    num_envs: int = 1
    spawn_seed_every: int = 100
    seeds_per_spawn: int = 10
    max_steps: int = 10000
    total_seeds_per_episode: int = 500
    step_cost: float = 0.0
    score_delta_reward: float = 1.0
    terminal_win_reward: float = 1.0
    policies: List[Any] = field(default_factory=list)


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _derive_batch_size(
    *,
    num_envs: int,
    requested_batch_size: Optional[int],
    device: str,
) -> int:
    batch_size = int(requested_batch_size) if requested_batch_size is not None else num_envs * 2
    batch_size = _round_up_to_multiple(max(num_envs, batch_size), num_envs)
    if device == "mps" and num_envs <= 2048:
        batch_size = min(batch_size, 2048)
        batch_size = _round_up_to_multiple(max(num_envs, batch_size), num_envs)
    return batch_size


def _derive_buffer_size(
    *,
    num_envs: int,
    num_agents: int,
    batch_size: int,
    requested_buffer_size: Optional[int],
) -> int:
    min_buffer_size = batch_size * num_agents
    default_buffer_size = min_buffer_size * 4
    buffer_size = (
        int(requested_buffer_size) if requested_buffer_size is not None else default_buffer_size
    )
    return _round_up_to_multiple(max(min_buffer_size, buffer_size), num_envs * num_agents)


def build_training_config(
    *,
    width: int = 20,
    height: int = 20,
    num_agents: int = 4,
    num_envs: int = 256,
    max_steps: int = 1000,
    episodes: int = 100,
    training_epochs: int = 1,
    lr: float = 0.001,
    clip_epsilon: float = 0.2,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    total_seeds: int = 500,
    batch_size: Optional[int] = None,
    buffer_size: Optional[int] = None,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    resolved_device = device or select_device()

    resolved_num_envs = max(1, int(num_envs))
    resolved_num_agents = max(1, int(num_agents))
    if resolved_device == "mps":
        resolved_num_envs = min(resolved_num_envs, 128)

    resolved_batch_size = _derive_batch_size(
        num_envs=resolved_num_envs,
        requested_batch_size=batch_size,
        device=resolved_device,
    )
    resolved_buffer_size = _derive_buffer_size(
        num_envs=resolved_num_envs,
        num_agents=resolved_num_agents,
        batch_size=resolved_batch_size,
        requested_buffer_size=buffer_size,
    )

    return {
        "width": width,
        "height": height,
        "num_agents": resolved_num_agents,
        "num_envs": resolved_num_envs,
        "max_steps": max_steps,
        "episodes": episodes,
        "training_epochs": training_epochs,
        "lr": lr,
        "clip_epsilon": clip_epsilon,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "total_seeds": total_seeds,
        "batch_size": resolved_batch_size,
        "buffer_size": resolved_buffer_size,
        "device": resolved_device,
        "dtype": dtype,
    }
"""

example_agent_code = """
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from simverse.abstractor.agent import SimAgent

if TYPE_CHECKING:
    from torch.nn import Module

DEFAULT_AGENT_ACTIONS = np.arange(6, dtype=np.int64)


class FarmtilaAgent(SimAgent):
    def __init__(
        self,
        agent_id: int,
        position: tuple[int, int],
        action_space: np.ndarray | None = None,
        policy: Optional["Module"] = None,
    ) -> None:
        action_space = action_space if action_space is not None else DEFAULT_AGENT_ACTIONS
        super().__init__(name=f"farmer_{agent_id}", action_space=action_space, policy=policy)
        self.agent_id = agent_id
        self.position = position
        self.inventory = 0
        self.harvested_tiles = 0
        self.reward = 0.0
        self.memory: dict = {}
        self._rng = np.random.default_rng(agent_id)

    def action(self, obs: np.ndarray) -> np.ndarray:
        if self.policy is not None:
            return self.policy(obs)
        return np.array([self._rng.choice(self.action_space)], dtype=np.int64)

    def info(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "position": self.position,
            "inventory": self.inventory,
            "harvested_tiles": self.harvested_tiles,
            "reward": self.reward,
        }

    def reset(self) -> None:
        self.inventory = 0
        self.harvested_tiles = 0
        self.reward = 0.0
        self.memory.clear()

    def get_action_space(self) -> np.ndarray:
        return self.action_space

    def get_memory(self) -> dict:
        return self.memory

    def current_state(self) -> np.ndarray:
        return np.array(
            [self.position[0], self.position[1], self.inventory, self.harvested_tiles],
            dtype=np.float32,
        )

    def get_policy(self):
        return self.policy

    def set_policy(self, policy) -> None:
        self.policy = policy
"""


def create_system_prompt(name: str) -> str:
    return (
        "You are a helpful agent who is an expert in building RL environments.\n"
        "You will converse with the user to understand their needs and build the "
        "environment accordingly.\n"
        "When the user provides all the details, use SimVerse abstractions to build "
        "the RL environment in its framework.\n"
        "You will write `env.py` with environment logic, `render.py` with render "
        "code, and `train.py` with training code.\n"
        "You can navigate to the `simverse-web` directory to reference existing code "
        "and build accordingly.\n"
        "Continue until the environment is built and training is working.\n"
        "After the environment is built, ask the user if they want to train the "
        "environment.\n"
        "If they say train, run training, show results, and return the results"
        "You have to generate the following code files: env.py, train.py, config.py, agent.py"
        f"example the following environment code: {example_env_code}"
        f"example train code: {example_train_code}"
        f"example config code: {example_config_code}"
        f"example agent code: {example_agent_code}"
        "This is a absctraction layer for RL environments for simverse."
        "You will be given a description of the environment and you will have to"
        "build the environment accordingly."
    )


class SimpleTerminalAgent:
    def __init__(
        self,
        name: str,
        workspace: Path,
        model: str = "gpt-5-nano",
    ) -> None:
        self.name = name
        self.workspace = workspace
        self.model = model
        self.client = self._build_client()
        self.history: list[dict[str, str]] = []

    def _build_client(self) -> OpenAI | None:
        if not os.getenv("OPENAI_API_KEY"):
            return None
        return OpenAI()

    def handle_user_message(self, user_input: str) -> AgentTurn:
        text = user_input.strip()
        if not text:
            return AgentTurn(status="idle", reply="Say something and I will reply.")

        if self.client is None:
            return AgentTurn(
                status="error",
                reply="OpenAI client unavailable. Install `openai` and set `OPENAI_API_KEY`.",
            )

        self.history.append({"role": "user", "content": text})

        messages = [{"role": "system", "content": create_system_prompt(self.name)}] + self.history[
            -20:
        ]
        response = self.client.responses.create(
            model=self.model,
            input=messages,
            reasoning={"effort": "minimal"},
            max_output_tokens=600,
        )
        reply = (getattr(response, "output_text", "") or "").strip()
        if not reply:
            reply = "I could not generate a response. Please try again."
            status = "error"
        else:
            status = "ok"

        self.history.append({"role": "assistant", "content": reply})
        return AgentTurn(status=status, reply=reply)


def create_agent(name: str, workspace: Path, model: str = "gpt-5-nano") -> SimpleTerminalAgent:
    return SimpleTerminalAgent(name=name, workspace=workspace, model=model)


def run_cli() -> None:
    agent = create_agent(name="SimVerse Assistant", workspace=Path(__file__).resolve().parent)
    print("Simple Terminal Agent")
    print("Type `exit` to stop.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        turn = agent.handle_user_message(user_input)
        print(f"Agent ({turn.status}): {turn.reply}\n")


if __name__ == "__main__":
    run_cli()
