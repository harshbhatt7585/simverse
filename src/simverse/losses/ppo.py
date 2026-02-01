from typing import Any, Dict, List, Optional

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
    BUFFER_SIZE = 10000
    BATCH_SIZE = 32

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
        self.replay_buffer = ReplayBuffer(self.BUFFER_SIZE)
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
        self.episode_save_dir = episode_save_dir
        self._env_metadata_cache: Dict[str, Any] | None = None

    def _get_optimizer(self, agent_id: int) -> torch.optim.Optimizer:
        if self.optimizers:
            if agent_id not in self.optimizers:
                raise KeyError(f"Missing optimizer for agent {agent_id}")
            return self.optimizers[agent_id]
        if self.optimizer is None:
            raise RuntimeError("No optimizer configured for PPOTrainer")
        return self.optimizer

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

    def _init_logging(self, title: str = "Training"):
        training_logger.header(title)
        if self.config:
            training_logger.config(self.config)

        if _WANDB_AVAILABLE:
            training_logger.info("Weights & Biases logging enabled")
            wandb.init(project=self.project_name, name=self.run_name, config=self.config)
            self._wandb_initialized = True
        else:
            training_logger.warning(
                "Weights & Biases not available - install with: pip install wandb"
            )

    def _finish_logging(self):
        if self._wandb_initialized and _WANDB_AVAILABLE:
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

        self._init_logging(title)
        training_logger.success("Environment and policies initialized")

        training_logger.start_training(self.episodes)

        for episode in range(self.episodes):
            training_logger.start_episode(episode + 1)
            self.stats.reset_episode()

            self.env.reset()
            episode_reward = 0.0

            for step in range(self.env.config.max_steps):
                obs = self.env.get_observation()

                ## INFERENCE PHASE (DATA COLLECTION)

                actions = {}
                collected_step_data = []
                for agent in self.agents:
                    agent.policy.eval()
                    with torch.no_grad():
                        # Extract observation tensor from dict and convert to torch
                        obs_tensor = torch.from_numpy(obs["obs"]).float().unsqueeze(0)
                        logits, value = agent.policy(obs_tensor)
                        dist = torch.distributions.Categorical(logits=logits)
                        action = dist.sample()
                        log_prob = dist.log_prob(action)
                        # Convert action to int for env.step
                        action_int = action.item()
                        actions[agent.agent_id] = action_int
                        collected_step_data.append(
                            {
                                "agent_id": agent.agent_id,
                                "observation": obs_tensor,
                                "action": action,
                                "log_prob": log_prob,
                                "value": value,
                            }
                        )

                obs, reward, done, info = self.env.step(actions)

                if self.episode_save_dir:
                    frame_record = self._build_frame_record(
                        obs,
                        actions,
                        reward,
                        info,
                        step + 1,
                        done,
                    )
                    self.stats.record_frame(frame_record)

                # store experiences for each agent
                for data in collected_step_data:
                    agent_reward = (
                        reward.get(data["agent_id"], 0.0) if isinstance(reward, dict) else reward
                    )
                    experience = Experience(
                        agent_id=data["agent_id"],
                        observation=data["observation"],
                        action=data["action"],
                        log_prob=data["log_prob"],
                        value=data["value"],
                        reward=agent_reward,
                        done=done,
                        info=info,
                    )
                    self.replay_buffer.add(experience)

                    # Track stats per agent
                    self.stats.push_experience(experience)
                self.stats.step()

                # Accumulate episode reward
                if isinstance(reward, dict):
                    episode_reward += sum(reward.values())
                else:
                    episode_reward += reward

                # Log step progress (every 10 steps to reduce output)
                if (step + 1) % 10 == 0 or step == self.env.config.max_steps - 1:
                    training_logger.log_step(
                        step + 1, self.env.config.max_steps, {"reward": episode_reward}
                    )

            # Clear the step progress line before training logs
            print()

            # TRAINING PHASE (MODEL UPDATE)
            for agent in self.agents:
                agent.policy.train()

                for epoch in range(self.training_epochs):
                    # Sample a batch of experiences (trajectory)
                    trajectory = self.replay_buffer.sample_for_agent(
                        agent.agent_id, self.BATCH_SIZE
                    )
                    if not trajectory:
                        break

                    # Extract trajectory data as lists
                    observations = [exp.observation for exp in trajectory]
                    actions = [exp.action for exp in trajectory]
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
                    advantages = self.compute_gae(rewards, values, next_value, dones)

                    # Compute returns (advantages + values)
                    returns = advantages + torch.tensor(values, dtype=torch.float32)

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

            avg_reward = episode_reward / max(self.env.config.max_steps, 1)
            training_logger.end_episode(
                episode + 1,
                total_reward=episode_reward,
                avg_reward=avg_reward,
                steps=self.env.config.max_steps,
            )

            self.stats.push_reward(episode_reward)

            if self.episode_save_dir:
                metadata = {
                    "env_config": self._env_metadata(),
                    "training_config": self.config,
                }
                output_path = self.stats.dump_episode_recording(
                    self.episode_save_dir,
                    episode + 1,
                    metadata=metadata,
                )
                training_logger.info(f"Saved episode metrics to {output_path}")

            self.save_checkpoint(f"checkpoints/ppo_checkpoint_{episode}.pth")

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
