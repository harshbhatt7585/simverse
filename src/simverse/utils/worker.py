import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict

from simverse.envs.farmtila.env import FarmtilaEnv
from simverse.policies.simple import SimplePolicy
from simverse.simulator import Simulator


@dataclass
class WorkerConfig:
    worker_id: int
    env_config: Dict[str, Any]
    policy_state: Dict[str, Any]


class Worker:
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.num_workers = config.num_workers
        self.envs = {
            worker_id: FarmtilaEnv(config.env_config)
            for worker_id in range(self.config.num_workers)
        }
        self.policy = SimplePolicy(config.policy_state)

    def run(self):
        with mp.Pool(processes=self.config.num_workers) as pool:
            results = pool.map(self._run_worker, range(self.config.num_workers))
        return results

    def _run_worker(self, worker_id: int) -> None:
        env = self.envs[worker_id]
        agents = self.env.agents
        Simulator(env, agents, self.policy)
        Simulator.run()
