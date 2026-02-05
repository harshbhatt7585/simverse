import time
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from simverse.abstractor.agent import SimAgent
from simverse.abstractor.simenv import SimEnv
from simverse.abstractor.trainer import Trainer
from simverse.agent.stats import TrainingStats
from simverse.logging_config import get_logger, training_logger
from simverse.utils.replay_buffer import Experience, ReplayBuffer

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False
logger = get_logger(__name__)


class PPOTrainer(Trainer):
    DEFAULT_BUFFER_SIZE = 10000
    DEFAULT_BATCH_SIZE = 32

    def __init__(
        self,
        optimizer: Optional[torch.optim.Optimizer] = None,
        optimizers: Optional[Dict[int, torch.optim.Optimizer]] = None,
        episodes: int = 1,
        training_epochs: int = 4,
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        stats: Optional[TrainingStats] = None,
        config: Optional[Dict[str, Any]] = None,
        project_name: str = "simverse",
        run_name: str = "ppo-training",
        episode_save_dir: str | None = None,
        device: Union[torch.device, str] = "cpu",
        batch_size: int = DEFAULT_BATCH_SIZE,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        dtype: torch.dtype = torch.float32,
        use_wandb: bool = True,
    ):
        super().__init__()

        if optimizer is None and not optimizers:
            raise ValueError(
                "PPOTrainer requires either a shared optimizer or per-agent optimizers"
            )
        if optimizer is not None and optimizers:
            raise ValueError("Provide only one of optimizer or optimizers")

        self.optimizer = optimizer
        self.optimizers = optimizers or {}
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.episodes = episodes
        self.training_epochs = training_epochs
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.stats = stats if stats is not None else TrainingStats()
        self.config = config or {}
        self.project_name = project_name
        self.run_name = run_name
        self._wandb_initialized = False
        self.use_wandb = use_wandb
        self.episode_save_dir = episode_save_dir
        self._env_metadata_cache: Dict[str, Any] | None = None
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.dtype = dtype
        self.env_batch_size = 1

    def _get_optimizer(self, agent_id: int) -> torch.optim.Optimizer:
        if self.optimizers:
            if agent_id not in self.optimizers:
                raise KeyError(f"Missing optimizer for agent {agent_id}")
            return self.optimizers[agent_id]
        if self.optimizer is None:
            raise RuntimeError("No optimizer configured for PPOTrainer")
        return self.optimizer

    def _move_policies_to_device(self) -> None:
        for agent in getattr(self, "agents", []):
            policy = getattr(agent, "policy", None)
            if policy is not None:
                policy.to(self.device)

    def _env_metadata(self) -> Dict[str, Any]:
        if self._env_metadata_cache is not None:
            return self._env_metadata_cache
        env = getattr(self, "env", None)
        config = getattr(env, "config", None)
        data: Dict[str, Any] = {}
        if config is not None:
            for attr in (
                "width",
                "height",
                "num_agents",
                "max_steps",
                "spawn_seed_every",
                "seeds_per_spawn",
                "total_seeds_per_episode",
            ):
                if hasattr(config, attr):
                    data[attr] = getattr(config, attr)
        self._env_metadata_cache = data
        return data

    def _format_rewards(self, rewards: Any) -> Any:
        if isinstance(rewards, dict):
            formatted = []
            for agent_id, value in rewards.items():
                try:
                    reward_value = float(value)
                except (TypeError, ValueError):
                    reward_value = 0.0
                formatted.append({"agent_id": agent_id, "reward": reward_value})
            return formatted
        try:
            return float(rewards)
        except (TypeError, ValueError):
            return rewards

    def _build_frame_record(
        self,
        observation: Dict[str, Any],
        actions: Dict[int, int],
        rewards: Any,
        info: Dict[str, Any],
        step: int,
        done: bool,
    ) -> Dict[str, Any]:
        obs_array = observation.get("obs")
        serialized_obs = obs_array.tolist() if hasattr(obs_array, "tolist") else obs_array
        return {
            "step": step,
            "observation": serialized_obs,
            "agents": observation.get("agents", []),
            "actions": [
                {"agent_id": agent_id, "action": action}
                for agent_id, action in sorted(actions.items())
            ],
            "rewards": self._format_rewards(rewards),
            "info": info,
            "done": bool(done),
        }

    def _obs_batch_array(self, observation: Dict[str, Any]) -> np.ndarray:
        obs_array = observation.get("obs")
        if isinstance(obs_array, np.ndarray):
            arr = obs_array
        else:
            arr = np.asarray(obs_array)
        if arr.ndim == 3:
            arr = np.expand_dims(arr, axis=0)
        return arr

    def _prepare_obs_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        obs_array = self._obs_batch_array(observation)
        return torch.from_numpy(obs_array).to(self.dtype).to(self.device)

    def _batch_size_from_obs(self, observation: Dict[str, Any]) -> int:
        return int(self._obs_batch_array(observation).shape[0])

    def _reward_to_array(self, reward: Any, batch_size: int) -> np.ndarray:
        if isinstance(reward, np.ndarray):
            return reward.astype(np.float32, copy=False)

        reward_array = np.zeros((batch_size, self.env.config.num_agents), dtype=np.float32)

        def _assign(row_idx: int, value: Any) -> None:
            if isinstance(value, dict):
                for agent_id, agent_reward in value.items():
                    reward_array[row_idx, int(agent_id)] = float(agent_reward)
            else:
                reward_array[row_idx, :] = float(value)

        if isinstance(reward, list):
            for row_idx, value in enumerate(reward[:batch_size]):
                _assign(row_idx, value)
        else:
            _assign(0, reward)

        return reward_array

    def _done_to_array(self, done: Any, batch_size: int) -> np.ndarray:
        if isinstance(done, np.ndarray):
            return done.astype(np.bool_, copy=False)
        if isinstance(done, (list, tuple)):
            return np.asarray(done, dtype=np.bool_)
        done_array = np.zeros(batch_size, dtype=np.bool_)
        done_array[:] = bool(done)
        return done_array

    def _ensure_info_list(self, info: Any, batch_size: int) -> List[Dict[str, Any]]:
        if isinstance(info, list):
            if len(info) == batch_size:
                return info
            if len(info) == 1:
                return info * batch_size
            padded = list(info)
            while len(padded) < batch_size:
                padded.append({})
            return padded[:batch_size]
        if isinstance(info, dict):
            return [dict(info) for _ in range(batch_size)]
        return [{} for _ in range(batch_size)]

    def _extract_env_observation(self, observation: Dict[str, Any], env_idx: int) -> Dict[str, Any]:
        obs_array = self._obs_batch_array(observation)
        env_obs = obs_array[env_idx]

        agents_field = observation.get("agents", [])
        env_agents: Any
        if agents_field and isinstance(agents_field[0], dict):
            env_agents = agents_field
        elif agents_field and env_idx < len(agents_field):
            env_agents = agents_field[env_idx]
        else:
            env_agents = []

        done_field = observation.get("done")
        if isinstance(done_field, (list, tuple, np.ndarray)):
            env_done = bool(done_field[env_idx])
        else:
            env_done = bool(done_field) if done_field is not None else False

        winner_field = observation.get("winner")
        if isinstance(winner_field, (list, tuple, np.ndarray)):
            env_winner = winner_field[env_idx]
        else:
            env_winner = winner_field

        steps_field = observation.get("steps")
        if isinstance(steps_field, (list, tuple, np.ndarray)):
            env_steps = int(steps_field[env_idx])
        else:
            env_steps = int(steps_field) if steps_field is not None else 0

        return {
            "obs": env_obs,
            "agents": env_agents,
            "done": env_done,
            "winner": env_winner,
            "steps": env_steps,
        }

    def _reward_row_to_dict(self, reward_row: np.ndarray) -> Dict[int, float]:
        return {
            agent_id: float(reward_row[agent_id]) for agent_id in range(self.env.config.num_agents)
        }

    def _init_logging(self, title: str = "Training"):
        training_logger.header(title)
        if self.config:
            training_logger.config(self.config)

        if self.use_wandb and _WANDB_AVAILABLE:
            training_logger.info("Weights & Biases logging enabled")
            wandb.init(project=self.project_name, name=self.run_name, config=self.config)
            self._wandb_initialized = True
        elif self.use_wandb:
            training_logger.warning(
                "Weights & Biases not available - install with: pip install wandb"
            )

    def _finish_logging(self):
        if self._wandb_initialized and self.use_wandb and _WANDB_AVAILABLE:
            wandb.finish()
            training_logger.success("Wandb run finished")

    # TODO: Looking suspicious, need to check if this is correct
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        next_value: float,
        dones: List[bool],
    ) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation for a trajectory.

        Args:
            rewards: List of rewards for each step
            values: List of value estimates for each step
            next_value: Value estimate for the final next state (bootstrap)
            dones: List of done flags for each step

        Returns:
            Tensor of advantages for each step
        """
        gae = 0.0
        advantages = []
        n_steps = len(rewards)

        for step in reversed(range(n_steps)):
            if step == n_steps - 1:
                next_val = next_value
            else:
                next_val = values[step + 1]

            done_mask = 1.0 - float(dones[step])
            delta = rewards[step] + self.gamma * next_val * done_mask - values[step]
            gae = delta + self.gamma * self.gae_lambda * done_mask * gae
            advantages.insert(0, gae)

        return torch.tensor(advantages, dtype=torch.float32)

    def train(
        self,
        env: SimEnv,
        agents: List[SimAgent],
        title: str = "Training",
    ):
        self.env = env
        self.agents = agents
        self._env_metadata_cache = None
        self._move_policies_to_device()

        self._init_logging(title)
        training_logger.success("Environment and policies initialized")

        training_logger.start_training(self.episodes)
        training_start = time.perf_counter()
        paused_time = 0.0
        last_active_time = 0.0
        total_agent_steps = 0
        last_logged_steps = 0

        for episode in range(self.episodes):
            training_logger.start_episode(episode + 1)
            self.stats.reset_episode()

            obs = self.env.reset()
            self.env_batch_size = self._batch_size_from_obs(obs)
            episode_reward = 0.0
            episode_agent_steps = 0

            for step in range(self.env.config.max_steps):
                obs_tensor = self._prepare_obs_tensor(obs)
                batch_envs = obs_tensor.shape[0]

                actions_per_env: List[Dict[int, int]] = [{} for _ in range(batch_envs)]
                collected_agent_data: Dict[int, Dict[str, torch.Tensor]] = {}

                for agent in self.agents:
                    agent.policy.eval()
                    with torch.no_grad():
                        logits, value = agent.policy(obs_tensor)
                        dist = torch.distributions.Categorical(logits=logits)
                        action = dist.sample()
                        log_prob = dist.log_prob(action)

                    collected_agent_data[agent.agent_id] = {
                        "action": action,
                        "log_prob": log_prob,
                        "value": value,
                    }

                    action_cpu = action.detach().cpu()
                    for env_idx in range(batch_envs):
                        actions_per_env[env_idx][agent.agent_id] = int(action_cpu[env_idx].item())

                env_actions: Union[Sequence[Dict[int, int]], Dict[int, int]]
                if batch_envs == 1:
                    env_actions = actions_per_env[0]
                else:
                    env_actions = actions_per_env

                obs, reward, done, info = self.env.step(env_actions)

                reward_array = self._reward_to_array(reward, batch_envs)
                done_array = self._done_to_array(done, batch_envs)
                info_list = self._ensure_info_list(info, batch_envs)

                if self.episode_save_dir:
                    for env_idx in range(batch_envs):
                        frame_obs = self._extract_env_observation(obs, env_idx)
                        frame_reward = self._reward_row_to_dict(reward_array[env_idx])
                        frame_record = self._build_frame_record(
                            frame_obs,
                            actions_per_env[env_idx],
                            frame_reward,
                            info_list[env_idx],
                            step + 1,
                            bool(done_array[env_idx]),
                        )
                        self.stats.record_frame(frame_record)

                for env_idx in range(batch_envs):
                    env_obs = obs_tensor[env_idx].unsqueeze(0).detach()
                    env_done = bool(done_array[env_idx])
                    env_info = info_list[env_idx]
                    for agent_id, agent_data in collected_agent_data.items():
                        reward_value = float(reward_array[env_idx, agent_id])
                        action_tensor = agent_data["action"][env_idx].unsqueeze(0).detach()
                        log_prob_tensor = agent_data["log_prob"][env_idx].unsqueeze(0).detach()
                        value_tensor = agent_data["value"][env_idx].unsqueeze(0).detach()
                        experience = Experience(
                            agent_id=agent_id,
                            observation=env_obs,
                            action=action_tensor,
                            log_prob=log_prob_tensor,
                            value=value_tensor,
                            reward=reward_value,
                            done=env_done,
                            info=env_info,
                        )
                        self.replay_buffer.add(experience)
                        self.stats.push_experience(experience)
                    self.stats.step()

                episode_reward += float(reward_array.sum())

                steps_this_iter = batch_envs * max(len(self.agents), 1)
                episode_agent_steps += steps_this_iter
                total_agent_steps += steps_this_iter

                if (step + 1) % 100 == 0 or step == self.env.config.max_steps - 1:
                    active_time = max(time.perf_counter() - training_start - paused_time, 1e-8)
                    delta_steps = total_agent_steps - last_logged_steps
                    delta_time = max(active_time - last_active_time, 1e-8)
                    steps_per_sec = delta_steps / delta_time
                    last_active_time = active_time
                    last_logged_steps = total_agent_steps
                    training_logger.log_step(
                        step + 1,
                        self.env.config.max_steps,
                        {
                            "reward": episode_reward,
                            "steps_per_sec": round(steps_per_sec, 2),
                        },
                    )

                if done_array.all():
                    break

            # Clear the step progress line before training logs
            print()

            # TRAINING PHASE (MODEL UPDATE)
            for agent in self.agents:
                agent.policy.train()

                for epoch in range(self.training_epochs):
                    # Sample a batch of experiences (trajectory)
                    trajectory = self.replay_buffer.sample_for_agent(
                        agent.agent_id, self.batch_size
                    )
                    if not trajectory:
                        break

                    # Extract trajectory data as lists
                    observations = [exp.observation for exp in trajectory]
                    rewards = [
                        sum(exp.reward.values()) if isinstance(exp.reward, dict) else exp.reward
                        for exp in trajectory
                    ]
                    values = [exp.value.squeeze().item() for exp in trajectory]
                    dones = [
                        exp.done if isinstance(exp.done, bool) else bool(exp.done)
                        for exp in trajectory
                    ]

                    # Get next value for bootstrap (from last observation)
                    with torch.no_grad():
                        _, next_value = agent.policy(observations[-1])
                        next_value = next_value.squeeze().item()

                    # Compute advantages for the trajectory
                    advantages = self.compute_gae(rewards, values, next_value, dones).to(
                        self.device
                    )

                    # Compute returns (advantages + values)
                    returns = advantages + torch.tensor(
                        values, dtype=self.dtype, device=self.device
                    )

                    # PPO update for each step in trajectory
                    for i, exp in enumerate(trajectory):
                        logits, value = agent.policy(exp.observation)
                        dist = torch.distributions.Categorical(logits=logits)
                        log_prob = dist.log_prob(exp.action)

                        ratio = torch.exp(log_prob - exp.log_prob)

                        adv = advantages[i]
                        surr1 = ratio * adv
                        surr2 = (
                            torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv
                        )
                        policy_loss = -torch.min(surr1, surr2).mean()

                        value_loss = 0.5 * (returns[i] - value.squeeze()).pow(2).mean()

                        loss = policy_loss + 0.5 * value_loss

                        optimizer = self._get_optimizer(agent.agent_id)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # Beautiful epoch logging
                    training_logger.log_epoch(
                        epoch, self.training_epochs, policy_loss.item(), value_loss.item()
                    )

                    # Track and log losses (per agent)
                    self.stats.push_agent_losses(
                        agent.agent_id, policy_loss.item(), value_loss.item()
                    )
                    self.stats.log_wandb(step=self.stats.steps)

            avg_reward = episode_reward / max(episode_agent_steps, 1)
            training_logger.end_episode(
                episode + 1,
                total_reward=episode_reward,
                avg_reward=avg_reward,
                steps=episode_agent_steps,
            )

            self.stats.push_reward(episode_reward)

            pause_start = time.perf_counter()
            if self.episode_save_dir:
                serializable_config = {
                    key: (str(value) if isinstance(value, torch.dtype) else value)
                    for key, value in self.config.items()
                }
                metadata = {
                    "env_config": self._env_metadata(),
                    "training_config": serializable_config,
                }
                output_path = self.stats.dump_episode_recording(
                    self.episode_save_dir,
                    episode + 1,
                    metadata=metadata,
                )
                training_logger.info(f"Saved episode metrics to {output_path}")

            self.save_checkpoint(f"checkpoints/ppo_checkpoint_{episode}.pth")
            paused_time += time.perf_counter() - pause_start

        training_logger.finish(
            {
                "avg_episode_reward": sum(self.stats.episode_rewards)
                / max(len(self.stats.episode_rewards), 1),
                "final_policy_loss": self.stats.policy_losses[-1]
                if self.stats.policy_losses
                else 0.0,
                "final_value_loss": self.stats.value_losses[-1] if self.stats.value_losses else 0.0,
            }
        )

        self._finish_logging()
