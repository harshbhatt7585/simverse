from __future__ import annotations

import multiprocessing as mp
from typing import Any, Dict

from simverse.envs.farmtila.config import FarmtilaConfig
from simverse.envs.farmtila.env import FarmtilaEnv
from simverse.policies.simple import SimplePolicy


def env_worker(worker_id: int, config: Dict[str, Any], data_queue: mp.Queue) -> None:
    """Runs a FarmtilaEnv and streams experiences back to the learner."""
    farm_config = FarmtilaConfig(**config["env"])
    env = FarmtilaEnv(farm_config)
    policy = SimplePolicy(env.observation_space, env.action_space)
    policy.load_state_dict(config["policy_state"])

    while True:
        obs = env.get_observation()
        # TODO: batch inference if policies are shared
        actions = {agent.agent_id: policy(agent) for agent in env.agents}
        next_obs, reward, done, info = env.step(actions)
        data_queue.put(
            {
                "worker_id": worker_id,
                "obs": obs,
                "actions": actions,
                "reward": reward,
                "done": done,
                "info": info,
                "steps": 1,
            }
        )
        if done:
            env.reset()
