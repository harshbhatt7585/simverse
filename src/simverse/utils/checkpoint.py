f"""
Each agent has their own policy,
To save the checkpoint of one training run on an ENV, we need to save all the policies of all the agents.
we can save like this:

farmtila_state_dict = {{
    agent_id -> policy_state_dict
}}


and load like this:

checkpointer.load(farmtila_state_dict_path)


def load_farmtila_state_dict(farmtila_state_dict_path: str) -> dict:
    with open(farmtila_state_dict_path, "rb") as f:
        farmtila_state_dict = pickle.load(f)
        for agent in agents:
            agent.policy.load_state_dict(farmtila_state_dict[agent.agent_id])

"""

import pickle
from simverse.abstractor.simenv import SimEnv

class Checkpointer:
    def __init__(self, env: SimEnv):
        self.env = env

    def save(self,  state_dict_path: str) -> None:
        agents = self.env.agents
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
            
