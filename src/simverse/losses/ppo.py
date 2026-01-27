import torch
import torch.nn as nn
from simverse.abstractor.trainer import Trainer
from simverse.abstractor.agent import SimAgent
from typing import List, Optional
import torch.nn.functional as F
from simverse.utils.replay_buffer import ReplayBuffer
from simverse.utils.replay_buffer import Experience
from simverse.abstractor.simenv import SimEnv
from simverse.agent.stats import TrainingStats
from simverse.logging_config import get_logger, training_logger

logger = get_logger(__name__)


class PPOTrainer(Trainer):

    BUFFER_SIZE = 10000
    BATCH_SIZE = 1



    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        episodes: int = 1,
        training_epochs: int = 4,
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        stats: Optional[TrainingStats] = None,
    ):
        super().__init__()

        self.optimizer = optimizer
        self.replay_buffer = ReplayBuffer(self.BUFFER_SIZE)
        self.episodes = episodes
        self.training_epochs = training_epochs
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.stats = stats if stats is not None else TrainingStats()

    
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
    ):
        self.env = env
        self.agents = agents
        
        # Start training with beautiful logger
        training_logger.start_training(self.episodes)
        
        for episode in range(self.episodes):
            training_logger.start_episode(episode + 1)
            
            self.env.reset()
            episode_reward = 0.0
            
            for step in range(self.env.config.max_steps):
                # observation of all the agents
                obs = self.env.get_observation()

                ## INFERENCE PHASE (DATA COLLECTION)

                # each agent has their own policy to take action
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
                        obs, reward, done, info = self.env.step({agent.agent_id: action_int})

                        # store the data into buffer
                        experience = Experience(
                            observation=obs_tensor,
                            action=action,
                            log_prob=log_prob,
                            value=value,
                            reward=reward,
                            done=done,
                            info=info
                        )
                        self.replay_buffer.add(experience)
                        
                        # Track stats
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
                        step + 1, 
                        self.env.config.max_steps,
                        {"reward": episode_reward}
                    )

            # Clear the step progress line before training logs
            print()
            
            # TRAINING PHASE (MODEL UPDATE)
            for agent in self.agents:
                agent.policy.train()

                for epoch in range(self.training_epochs):
                    # Sample a batch of experiences (trajectory)
                    trajectory = self.replay_buffer.sample(self.BATCH_SIZE)
                    if not trajectory:
                        break
                    
                    # Extract trajectory data as lists
                    observations = [exp.observation for exp in trajectory]
                    actions = [exp.action for exp in trajectory]
                    old_log_probs = [exp.log_prob for exp in trajectory]
                    rewards = [sum(exp.reward.values()) if isinstance(exp.reward, dict) else exp.reward for exp in trajectory]
                    values = [exp.value.squeeze().item() for exp in trajectory]
                    dones = [exp.done if isinstance(exp.done, bool) else bool(exp.done) for exp in trajectory]
                    
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
                        
                        # Ratio for PPO
                        ratio = torch.exp(log_prob - exp.log_prob)
                        
                        # Clipped surrogate objective
                        adv = advantages[i]
                        surr1 = ratio * adv
                        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # Value loss
                        value_loss = 0.5 * (returns[i] - value.squeeze()).pow(2).mean()
                        
                        # Total loss
                        loss = policy_loss + 0.5 * value_loss
                        
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    
                    # Beautiful epoch logging
                    training_logger.log_epoch(
                        epoch, 
                        self.training_epochs, 
                        policy_loss.item(), 
                        value_loss.item()
                    )
                    
                    # Track and log losses
                    self.stats.push_losses(policy_loss.item(), value_loss.item())
                    self.stats.log_wandb(step=self.stats.steps)
            
            # Episode summary
            avg_reward = episode_reward / max(self.env.config.max_steps, 1)
            training_logger.end_episode(
                episode + 1,
                total_reward=episode_reward,
                avg_reward=avg_reward,
                steps=self.env.config.max_steps
            )
            
            # Track episode reward
            self.stats.push_reward(episode_reward)
        
        # Final summary
        training_logger.finish({
            "avg_episode_reward": sum(self.stats.episode_rewards) / max(len(self.stats.episode_rewards), 1),
            "final_policy_loss": self.stats.policy_losses[-1] if self.stats.policy_losses else 0.0,
            "final_value_loss": self.stats.value_losses[-1] if self.stats.value_losses else 0.0,
        })








            
                

            







