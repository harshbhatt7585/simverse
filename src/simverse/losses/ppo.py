import torch
import torch.nn as nn
from simverse.abstractor.trainer import Trainer
from simverse.abstractor.agent import SimAgent
from typing import List
import torch.nn.functional as F
from simverse.utils.replay_buffer import ReplayBuffer
from simverse.utils.replay_buffer import Experience



class PPOTrainer(Trainer):

    BUFFER_SIZE = 10000
    BATCH_SIZE = 1



    def __init__(
        self,
        env,
        agents: List[SimAgent],
        optimizer: torch.optim.Optimizer,
    ):
        super().__init__()
        self.env = env
        self.agents = agents

        self.optimizer = optimizer
        self.replay_buffer = ReplayBuffer(self.BUFFER_SIZE)


    
    def train(
        self
    ):
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
                        logits, value = agent.policy(obs) # this will call the neural net (policy) to compute the logits and value
                        dist = torch.distributions.Categorical(logits)
                        action = dist.sample()
                        log_prob = dist.log_prob(action)
                        obs, reward, done, info = self.env.step(action)

                        # store the data into buffer
                        self.replay_buffer.add(
                            Experience(
                                observation=obs,
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

                # sample the data from buffer
                batch = self.replay_buffer.sample(self.BATCH_SIZE)

                (_obs, _action, _log_prob, _value, reward, _, _info) = batch

                # get the action logits and value
                logits, value = agent.policy(obs)
                dist = torch.distributions.Categorical(logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)



                # compute the loss
                advantage = self.compute_gae(reward, value)

                ratio = torch.exp(log_prob - _log_prob)
                surr = ratio * advantage
                surr_clipped = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
                ppo_loss = -torch.min(surr, surr_clipped).mean()

                self.optimizer.zero_grad()

                ppo_loss.backward()
                self.optimizer.step()








            
                

            










