from dataclasses import dataclass
from typing import Any, Dict, List

from simverse.envs.farmtila.env import FarmtilaEnv
from simverse.policies.simple import SimplePolicy


@dataclass
class WorkerConfig:
    worker_id: int
    env_config: Dict[str, Any]
    policy_state: Dict[str, Any]


class Worker:
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.env = FarmtilaEnv(config.env_config)
        self.policy = SimplePolicy(config.policy_state)

    def _build_envs(self, num_envs: int) -> List[FarmtilaEnv]:
        return [FarmtilaEnv(self.config.env_config) for _ in range(num_envs)]
