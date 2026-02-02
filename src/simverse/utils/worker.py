from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class WorkerConfig:
    worker_id: int
    env_config: Dict[str, Any]
    policy_state: Dict[str, Any]


class Worker:
    def __init__(self, config: WorkerConfig):
        self.config = config
