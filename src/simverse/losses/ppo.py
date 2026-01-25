import torch
import torch.nn as nn
from simverse.abstractor.trainer import Trainer
from simverse.abstractor.agent import SimAgent
from typing import List
import torch.nn.functional as F
from simverse.utils.replay_buffer import ReplayBuffer



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

        self.env.reset()
        for i in range(self.env.config.max_steps):

            # observation of all the agents
            obs = self.env.get_observation()


            ## INFERENCE PHASE (DATA COLLECTION)

            # each agent has their own policy to take action
            for agent in self.agents:
                agent.policy.eval()
                action = agent.action(obs) # this will call the neural net (policy) to compute the logits and value
                obs, reward, done, info = self.env.step(action)

                # store the data into buffer
                self.replay_buffer.add((obs, action, reward, done, info))
                
                # update the agent's memory
            

            # TRAINING PHASE (MODEL UPDATE)
            for agent in self.agents:
                agent.policy.train()

                # sample the data from buffer
                batch = sample_batch(buffer)

                (_obs, _action, reward, _, _info) = batch

                # get the action logits and value
                logits, value = agent.policy(obs)

                # get the action
                log_probs = F.log_softmax(logits, dim=1)
                action = torch.argmax(logits, dim=1)


                # compute the loss
                advantage = self.compute_gae(reward, value)

                ppo_loss = self.compute_ppo_loss(
                    log_probs,
                    action,
                    advantage,
                    value
                )

                self.optimizer.zero_grad()

                ppo_loss.backward()
                self.optimizer.step()








            
                

            










