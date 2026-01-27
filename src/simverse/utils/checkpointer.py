import os
import pickle
from simverse.abstractor.simenv import SimEnv

class Checkpointer:
    def __init__(self, env: SimEnv):
        self.env = env

    def save(self,  state_dict_path: str) -> None:
        agents = self.env.agents
        directory = os.path.dirname(state_dict_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        farmtila_state_dict = {
            "env_config": self.env.config,
            "agents": [
                {
                    "agent_id": agent.agent_id,
                    "policy_state_dict": agent.policy.state_dict()
                }
                for agent in agents
            ],
            "steps": self.env.steps,
        }
        with open(state_dict_path, "wb") as f:
            pickle.dump(farmtila_state_dict, f)

    
    def load(self, state_dict_path: str) -> None:
        with open(state_dict_path, "rb") as f:
            farmtila_state_dict = pickle.load(f)
            self.env.config = farmtila_state_dict["env_config"]
            for agent in farmtila_state_dict["agents"]:
                self.env.agents[agent["agent_id"]].policy.load_state_dict(agent["policy_state_dict"])
            
