import time
from typing import Any, Dict, List, Optional, Union

import torch
from simverse.abstractor.agent import SimAgent
from simverse.abstractor.simenv import SimEnv
from simverse.abstractor.trainer import Trainer
from simverse.agent.stats import TrainingStats
from simverse.logging_config import get_logger, training_logger
from simverse.utils.parallel_env_runner import ParallelEnvRunner
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
        use_parallel_env: bool = False,
        parallel_env_workers: int = 4,
        parallel_env_queue_size: int = 0,
        parallel_env_start_method: Optional[str] = None,
        parallel_env_warmup_steps: Optional[int] = None,
        parallel_env_steps_per_iteration: Optional[int] = None,
        parallel_env_timeout: float = 5.0,
        parallel_env_device: Optional[str] = None,
        parallel_env_dtype: Optional[torch.dtype] = None,
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
        warmup_default = max(buffer_size // 4, batch_size)
        steps_default = max(batch_size, 1)
        warmup_value = (
            warmup_default if parallel_env_warmup_steps is None else parallel_env_warmup_steps
        )
        steps_value = (
            steps_default
            if parallel_env_steps_per_iteration is None
            else parallel_env_steps_per_iteration
        )
        self.use_parallel_env = use_parallel_env
        self.parallel_env_workers = max(1, parallel_env_workers)
        self.parallel_env_queue_size = parallel_env_queue_size
        self.parallel_env_start_method = parallel_env_start_method
        self.parallel_env_timeout = parallel_env_timeout
        self.parallel_env_device = parallel_env_device or "cpu"
        self.parallel_env_dtype = parallel_env_dtype or dtype
        self.parallel_env_warmup_steps = max(0, warmup_value)
        self.parallel_env_steps_per_iteration = max(1, steps_value)

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

    def _prepare_observation(self, observation: Any) -> torch.Tensor:
        if isinstance(observation, torch.Tensor):
            obs_tensor = observation.detach()
        else:
            obs_tensor = torch.as_tensor(observation, dtype=self.dtype)
        if obs_tensor.dim() == 3:
            obs_tensor = obs_tensor.unsqueeze(0)
        return obs_tensor.to(self.device, dtype=self.dtype)

    def _prepare_action(self, action: Any) -> torch.Tensor:
        if isinstance(action, torch.Tensor):
            action_tensor = action.detach()
        else:
            action_tensor = torch.as_tensor(action, dtype=torch.long)
        return action_tensor.to(self.device)

    def _prepare_log_prob(self, log_prob: Any) -> torch.Tensor:
        if isinstance(log_prob, torch.Tensor):
            log_prob_tensor = log_prob.detach()
        else:
            log_prob_tensor = torch.as_tensor(log_prob, dtype=self.dtype)
        return log_prob_tensor.to(self.device, dtype=self.dtype)

    def _value_to_float(self, value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().squeeze().item())
        if hasattr(value, "item"):
            try:
                return float(value.item())
            except (TypeError, ValueError):
                return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _done_to_bool(self, done: Any) -> bool:
        if isinstance(done, torch.Tensor):
            return bool(done.detach().cpu().item())
        return bool(done)

    def _extract_reward(self, reward: Any) -> float:
        if isinstance(reward, dict):
            total = 0.0
            for value in reward.values():
                try:
                    total += float(value)
                except (TypeError, ValueError):
                    continue
            return total
        try:
            return float(reward)
        except (TypeError, ValueError):
            return 0.0

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

        if self.use_parallel_env:
            self._train_with_parallel_env_loop()
            return

        training_logger.start_training(self.episodes)
        training_start = time.perf_counter()
        paused_time = 0.0
        last_active_time = 0.0
        last_total_steps = 0

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
                        obs_tensor = (
                            torch.from_numpy(obs["obs"])  # type: ignore[arg-type]
                            .to(self.dtype)
                            .unsqueeze(0)
                            .to(self.device)
                        )
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
                    active_time = max(time.perf_counter() - training_start - paused_time, 1e-8)
                    total_steps_done = episode * self.env.config.max_steps + step + 1
                    delta_steps = total_steps_done - last_total_steps
                    delta_time = max(active_time - last_active_time, 1e-8)
                    steps_per_sec = delta_steps / delta_time
                    last_active_time = active_time
                    last_total_steps = total_steps_done
                    training_logger.log_step(
                        step + 1,
                        self.env.config.max_steps,
                        {
                            "reward": episode_reward,
                            "steps_per_sec": round(steps_per_sec, 2),
                        },
                    )

            # Clear the step progress line before training logs
            print()

            # TRAINING PHASE (MODEL UPDATE)
            self._train_from_replay_buffer()

            avg_reward = episode_reward / max(self.env.config.max_steps, 1)
            training_logger.end_episode(
                episode + 1,
                total_reward=episode_reward,
                avg_reward=avg_reward,
                steps=self.env.config.max_steps,
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

    def _train_from_replay_buffer(self) -> None:
        for agent in self.agents:
            policy = getattr(agent, "policy", None)
            if policy is None:
                continue
            policy.train()

            for epoch in range(self.training_epochs):
                trajectory = self.replay_buffer.sample_for_agent(agent.agent_id, self.batch_size)
                if not trajectory:
                    break

                observations = [self._prepare_observation(exp.observation) for exp in trajectory]
                actions = [self._prepare_action(exp.action) for exp in trajectory]
                rewards = [self._extract_reward(exp.reward) for exp in trajectory]
                values = [self._value_to_float(exp.value) for exp in trajectory]
                dones = [self._done_to_bool(exp.done) for exp in trajectory]
                log_probs_old = [self._prepare_log_prob(exp.log_prob) for exp in trajectory]

                with torch.no_grad():
                    _, next_value_tensor = policy(observations[-1])
                    next_value = self._value_to_float(next_value_tensor)

                advantages = self.compute_gae(rewards, values, next_value, dones).to(self.device)
                returns = advantages + torch.tensor(values, dtype=self.dtype, device=self.device)

                optimizer = self._get_optimizer(agent.agent_id)
                policy_loss_value = torch.tensor(0.0, device=self.device)
                value_loss_value = torch.tensor(0.0, device=self.device)

                for i in range(len(trajectory)):
                    logits, value = policy(observations[i])
                    dist = torch.distributions.Categorical(logits=logits)
                    log_prob = dist.log_prob(actions[i])
                    ratio = torch.exp(log_prob - log_probs_old[i])

                    adv = advantages[i]
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv
                    policy_loss_value = -torch.min(surr1, surr2).mean()

                    value_loss_value = 0.5 * (returns[i] - value.squeeze()).pow(2).mean()
                    loss = policy_loss_value + 0.5 * value_loss_value

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                training_logger.log_epoch(
                    epoch, self.training_epochs, policy_loss_value.item(), value_loss_value.item()
                )
                self.stats.push_agent_losses(
                    agent.agent_id, policy_loss_value.item(), value_loss_value.item()
                )
                self.stats.log_wandb(step=self.stats.steps)

    def _collect_policy_states(self) -> Dict[int, Dict[str, Any]]:
        states: Dict[int, Dict[str, Any]] = {}
        for agent in getattr(self, "agents", []):
            policy = getattr(agent, "policy", None)
            if policy is None:
                continue
            states[agent.agent_id] = policy.state_dict()
        if not states:
            raise RuntimeError("No policy states available for parallel env runner")
        return states

    def _serialize_env_config(self) -> Dict[str, Any]:
        env = getattr(self, "env", None)
        if env is None:
            raise RuntimeError("Trainer has no environment bound")
        config = getattr(env, "config", None)
        if config is None:
            raise RuntimeError("Environment is missing configuration for parallel runner")
        config_dict = dict(vars(config))
        config_dict.pop("policies", None)
        return config_dict

    def _train_with_parallel_env_loop(self) -> None:
        runner = ParallelEnvRunner(
            num_workers=self.parallel_env_workers,
            env_config=self._serialize_env_config(),
            policy_state=self._collect_policy_states(),
            device=self.parallel_env_device,
            dtype=self.parallel_env_dtype,
            queue_maxsize=self.parallel_env_queue_size,
            start_method=self.parallel_env_start_method,
        )

        training_logger.info(
            "Starting parallel collector with %s workers (start=%s)"
            % (self.parallel_env_workers, runner.start_method)
        )

        payload_counter = {"count": 0, "transitions": 0}
        rate_tracker = {
            "last_log_time": time.perf_counter(),
            "last_transitions": 0,
            "last_payloads": 0,
        }
        phase_state = {
            "active": False,
            "target": 1,
            "collected": 0,
            "label": "Collector",
            "last_ratio": -0.05,
        }

        def _start_collection_phase(target: int, label: str) -> None:
            if target <= 0:
                phase_state["active"] = False
                return
            phase_state["active"] = True
            phase_state["target"] = max(1, target)
            phase_state["collected"] = 0
            phase_state["label"] = label
            phase_state["last_ratio"] = -0.05
            rate_tracker["last_log_time"] = time.perf_counter()
            rate_tracker["last_payloads"] = payload_counter["count"]
            rate_tracker["last_transitions"] = payload_counter["transitions"]
            training_logger.info(f"{label}: collecting {phase_state['target']} agent-steps")

        def payload_hook(payload: Dict[str, Any]) -> None:
            self.stats.step()
            payload_counter["count"] += 1
            transitions = len(payload.get("collected_data", []))
            payload_counter["transitions"] += transitions
            worker_id = payload.get("worker_id")
            episode_step = payload.get("episode_step")
            done = payload.get("done")
            if phase_state["active"]:
                phase_state["collected"] += transitions
                now = time.perf_counter()
                elapsed = max(now - rate_tracker["last_log_time"], 1e-6)
                payload_delta = payload_counter["count"] - rate_tracker["last_payloads"]
                trans_delta = payload_counter["transitions"] - rate_tracker["last_transitions"]
                ratio = phase_state["collected"] / max(phase_state["target"], 1)
                if ratio >= 1.0 or ratio - phase_state["last_ratio"] >= 0.05 or elapsed >= 2.0:
                    training_logger.log_step(
                        min(int(phase_state["collected"]), int(phase_state["target"])),
                        int(phase_state["target"]),
                        {
                            "agent_steps/s": trans_delta / elapsed,
                            "payloads/s": payload_delta / elapsed,
                        },
                    )
                    phase_state["last_ratio"] = ratio
                    rate_tracker["last_log_time"] = now
                    rate_tracker["last_payloads"] = payload_counter["count"]
                    rate_tracker["last_transitions"] = payload_counter["transitions"]
            if done:
                training_logger.info(
                    "Collector episode done: worker=%s steps=%s" % (worker_id, episode_step)
                )

        def collect_transitions(target: int, label: str) -> None:
            if target <= 0:
                return
            _start_collection_phase(target, label)
            try:
                runner.pump_replay_buffer(
                    buffer=self.replay_buffer,
                    min_transitions=target,
                    timeout=self.parallel_env_timeout,
                    on_experience=self.stats.push_experience,
                    on_payload=payload_hook,
                )
            finally:
                phase_state["active"] = False
                print()

        training_logger.start_training(self.episodes)
        with runner:
            if self.parallel_env_warmup_steps > 0:
                collect_transitions(
                    self.parallel_env_warmup_steps,
                    "Warmup collection",
                )
                self.stats.reset_episode()

            for episode in range(self.episodes):
                training_logger.start_episode(episode + 1)
                self.stats.reset_episode()

                collect_transitions(
                    self.parallel_env_steps_per_iteration,
                    f"Episode {episode + 1} rollout",
                )

                episode_reward = sum(self.stats.step_rewards)
                avg_reward = (
                    episode_reward / max(len(self.stats.step_rewards), 1)
                    if self.stats.step_rewards
                    else 0.0
                )
                self.stats.push_reward(episode_reward)

                self._train_from_replay_buffer()
                runner.set_policy_state(self._collect_policy_states(), restart=True)

                training_logger.end_episode(
                    episode + 1,
                    total_reward=episode_reward,
                    avg_reward=avg_reward,
                    steps=self.parallel_env_steps_per_iteration,
                )
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
