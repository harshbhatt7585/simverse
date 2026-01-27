import torch
import torch.nn as nn
from simverse.abstractor.trainer import Trainer
from simverse.abstractor.agent import SimAgent
from typing import List
import torch.nn.functional as F
from simverse.utils.replay_buffer import ReplayBuffer
from simverse.utils.replay_buffer import Experience
from simverse.abstractor.simenv import SimEnv


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
    ):
        super().__init__()

        self.optimizer = optimizer
        self.replay_buffer = ReplayBuffer(self.BUFFER_SIZE)
        self.episodes = episodes
        self.training_epochs = training_epochs
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    
    def compute_gae(self, rewards, values, next_values, dones):
        gae = 0.0
        advantages = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_values[step] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        return advantages


    
    def train(
        self,
        env: SimEnv,
        agents: List[SimAgent],
    ):
        self.env = env
        self.agents = agents
        for _ in range(self.episodes):

            self.env.reset()
            for i in range(self.env.config.max_steps):

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
                        self.replay_buffer.add(
                            Experience(
                                observation=obs_tensor,
                                action=action,
                                log_prob=log_prob,
                                value=value,
                                reward=reward,
                                done=done,
                                info=info
                        ))
                    
            

            # TRAINING PHASE (MODEL UPDATE)
            for agent in self.agents:
                agent.policy.train()

                for epoch in range(self.training_epochs):
                    minibatch = self.replay_buffer.sample(self.BATCH_SIZE)
                    print("minibatch: ", minibatch)
                    if not minibatch:
                        break
                
                    

                    for experience in minibatch:
                        _obs = experience.observation
                        _action = experience.action
                        _log_prob = experience.log_prob
                        reward = experience.reward
                        _value = experience.value
                        _done = experience.done

                        logits, next_value = agent.policy(_obs)
                        dist = torch.distributions.Categorical(logits=logits)
                        log_prob = dist.log_prob(_action)


                        advantage = self.compute_gae(reward, _value, next_value, _done)

                        ratio = torch.exp(log_prob - _log_prob)
                        surr = ratio * advantage
                        surr_clipped = (
                            torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                            * advantage
                        )
                        ppo_loss = -torch.min(surr, surr_clipped).mean()
                        print(ppo_loss)
                        print(ppo_loss)

                        self.optimizer.zero_grad()
                        ppo_loss.backward()
                        self.optimizer.step()








            
                

            







