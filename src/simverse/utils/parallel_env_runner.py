from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional

import torch
from simverse.envs.farmtila.config import FarmtilaConfig
from simverse.envs.farmtila.env import FarmtilaEnv
from simverse.policies.simple import SimplePolicy
from simverse.utils.replay_buffer import Experience, ReplayBuffer
from simverse.utils.worker import env_worker


def _resolve_start_method(preferred: Optional[str] = None) -> str:
    supported = mp.get_all_start_methods()
    if preferred and preferred in supported:
        return preferred
    if "fork" in supported:
        return "fork"
    return supported[0]


def _detach_state_dict(state_dict: Any) -> Any:
    if isinstance(state_dict, torch.Tensor):
        return state_dict.detach().cpu()
    if isinstance(state_dict, dict):
        return {key: _detach_state_dict(value) for key, value in state_dict.items()}
    if isinstance(state_dict, list):
        return [_detach_state_dict(item) for item in state_dict]
    return state_dict


class ParallelEnvRunner:
    """Manages a fleet of Farmtila env workers and feeds a replay buffer."""

    def __init__(
        self,
        *,
        num_workers: int = 24,
        env_config: Optional[Dict[str, Any]] = None,
        policy_state: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        queue_maxsize: int = 0,
        start_method: Optional[str] = None,
    ) -> None:
        self.num_workers = num_workers
        self.env_config = env_config or {}
        self.device = device
        self.dtype = dtype
        self.start_method = _resolve_start_method(start_method)
        self.ctx = mp.get_context(self.start_method)
        self.data_queue: mp.Queue = self.ctx.Queue(maxsize=queue_maxsize)
        self.policy_state = self._init_policy_state(policy_state)
        self.processes: List[mp.Process] = []
        self._started = False

    def _init_policy_state(self, policy_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if policy_state is not None:
            return _detach_state_dict(policy_state)
        env = FarmtilaEnv(FarmtilaConfig(**self.env_config))
        policy = SimplePolicy(env.observation_space, env.action_space)
        return _detach_state_dict(policy.state_dict())

    def start(self) -> None:
        if self._started:
            return
        worker_config = {"env": self.env_config, "policy_state": self.policy_state}
        for worker_id in range(self.num_workers):
            proc = self.ctx.Process(
                target=env_worker,
                args=(worker_id, worker_config, self.data_queue, self.device, self.dtype),
            )
            proc.daemon = True
            proc.start()
            self.processes.append(proc)
        self._started = True

    def stop(self, *, timeout: float = 2.0) -> None:
        for proc in self.processes:
            if proc.is_alive():
                proc.terminate()
        for proc in self.processes:
            if proc.is_alive():
                proc.join(timeout=timeout)
            if proc.is_alive():
                proc.kill()
        self.processes.clear()
        self._started = False

    def restart(self) -> None:
        self.stop()
        self.start()

    def set_policy_state(self, state_dict: Dict[str, Any], *, restart: bool = False) -> None:
        self.policy_state = _detach_state_dict(state_dict)
        if restart and self._started:
            self.restart()

    def update_policy(self, policy: SimplePolicy, *, restart: bool = False) -> None:
        self.set_policy_state(policy.state_dict(), restart=restart)

    def _next_payload(self, timeout: float | None = None) -> Dict[str, Any]:
        payload = self.data_queue.get(timeout=timeout)
        if "error" in payload:
            worker_id = payload.get("worker_id")
            error = payload.get("error")
            error_type = payload.get("error_type")
            raise RuntimeError(f"Worker {worker_id} crashed: {error} ({error_type})")
        return payload

    def _payload_to_experiences(self, payload: Dict[str, Any]) -> List[Experience]:
        experiences: List[Experience] = []
        rewards = payload.get("reward", {})
        done_flag = payload.get("done", False)
        info = payload.get("info", {}) or {}
        worker_id = payload.get("worker_id")
        for agent_data in payload.get("collected_data", []):
            agent_id = agent_data.get("agent_id")
            reward_value = rewards.get(agent_id, 0.0) if isinstance(rewards, dict) else rewards
            info_copy = dict(info)
            if worker_id is not None:
                info_copy.setdefault("worker_id", worker_id)
            experiences.append(
                Experience(
                    agent_id=agent_id,
                    observation=self._to_cpu(agent_data.get("observation")),
                    action=self._to_cpu(agent_data.get("action")),
                    log_prob=self._to_cpu(agent_data.get("log_prob")),
                    value=self._to_cpu(agent_data.get("value")),
                    reward=reward_value,
                    done=done_flag,
                    info=info_copy,
                )
            )
        return experiences

    @staticmethod
    def _to_cpu(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        return value

    def collect(
        self, *, max_items: Optional[int] = None, timeout: float = 5.0
    ) -> List[Dict[str, Any]]:
        if not self._started:
            self.start()
        payloads: List[Dict[str, Any]] = []
        while True:
            payload = self._next_payload(timeout=timeout)
            payloads.append(payload)
            if max_items is not None and len(payloads) >= max_items:
                break
            if max_items is None:
                break
        return payloads

    def pump_replay_buffer(
        self,
        buffer: ReplayBuffer,
        *,
        min_transitions: int,
        timeout: float = 5.0,
        on_experience: Optional[Callable[[Experience], None]] = None,
        on_payload: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> int:
        if not self._started:
            self.start()
        added = 0
        while added < min_transitions:
            payload = self._next_payload(timeout=timeout)
            if on_payload is not None:
                on_payload(payload)
            experiences = self._payload_to_experiences(payload)
            if on_experience is not None:
                for exp in experiences:
                    on_experience(exp)
            added += buffer.extend(experiences)
        return added

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()


def warm_start_buffer(
    *,
    buffer: ReplayBuffer,
    num_transitions: int,
    num_workers: int = 24,
    env_config: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> int:
    runner = ParallelEnvRunner(
        num_workers=num_workers,
        env_config=env_config,
        device=device,
        dtype=dtype,
    )
    with runner:
        return runner.pump_replay_buffer(buffer, min_transitions=num_transitions)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parallel Farmtila data collector")
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--transitions", type=int, default=500)
    args = parser.parse_args()

    rb = ReplayBuffer(max_size=args.transitions * 2)
    collected = warm_start_buffer(
        buffer=rb,
        num_transitions=args.transitions,
        num_workers=args.num_workers,
    )
    buffer_size = len(rb)
    print(
        "Collected"
        f" {collected}"
        f" transitions using {args.num_workers} workers. Buffer size: {buffer_size}"
    )
