from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

if __package__ is None or __package__.startswith("__main__"):
    _src = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_src))

import torch
from simverse.envs.farmtila.config import FarmtilaConfig
from simverse.envs.farmtila.env import FarmtilaEnv
from simverse.policies.simple import SimplePolicy
from torch.distributions import Categorical


def _is_multi_policy_state(policy_state: Any) -> bool:
    if not isinstance(policy_state, Mapping):
        return False
    if not policy_state:
        return False
    return all(isinstance(key, int) for key in policy_state.keys())


def _init_policies(
    env: FarmtilaEnv,
    policy_state: Any,
    device: str,
) -> SimplePolicy | Dict[int, SimplePolicy]:
    if _is_multi_policy_state(policy_state):
        # Ensure deterministic fallback if a specific agent state is missing
        default_state = next(iter(policy_state.values()), None)
        policies: Dict[int, SimplePolicy] = {}
        for agent in env.agents:
            model = SimplePolicy(env.observation_space, env.action_space)
            state = policy_state.get(agent.agent_id, default_state)
            if state is not None:
                model.load_state_dict(state)
            model.to(device)
            model.eval()
            policies[agent.agent_id] = model
        return policies

    policy = SimplePolicy(env.observation_space, env.action_space)
    if policy_state:
        policy.load_state_dict(policy_state)
    policy.to(device)
    policy.eval()
    return policy


def env_worker(
    worker_id: int,
    config: Dict[str, Any],
    data_queue: mp.Queue,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> None:
    """
    Runs a FarmtilaEnv and streams experiences back to the learner.

    Args:
        worker_id: Unique identifier for this worker
        config: Configuration dictionary containing:
            - "env": Environment configuration dict for FarmtilaConfig
            - "policy_state": State dict of the policy to load
        data_queue: Queue to send experiences to the learner process
        device: Device to run inference on ("cpu" or "mps")
        dtype: Data type for tensors
    """
    try:
        farm_config = FarmtilaConfig(**config["env"])
        env = FarmtilaEnv(farm_config)
        policy_state = config.get("policy_state")
        obs = env.reset()
        policies = _init_policies(env, policy_state, device)
        episode_step = 0
        while True:
            current_obs = obs
            actions = {}
            collected_data = []
            obs_array = current_obs["obs"]
            obs_tensor = torch.from_numpy(obs_array).to(dtype)
            obs_batch = obs_tensor.unsqueeze(0)
            num_agents = len(env.agents)
            if isinstance(policies, dict):
                obs_on_device = obs_batch.to(device)
                for agent in env.agents:
                    policy = policies.get(agent.agent_id)
                    if policy is None:
                        continue
                    with torch.no_grad():
                        logits, value = policy(obs_on_device)
                        dist = Categorical(logits=logits)
                        action = dist.sample()
                        log_prob = dist.log_prob(action)
                    actions[agent.agent_id] = action.item()
                    collected_data.append(
                        {
                            "agent_id": agent.agent_id,
                            "observation": obs_batch,
                            "action": action,
                            "log_prob": log_prob,
                            "value": value,
                        }
                    )
            elif num_agents > 1:
                obs_expanded = obs_batch.repeat(num_agents, 1, 1, 1).to(device)
                with torch.no_grad():
                    logits_batch, values_batch = policies(obs_expanded)
                    dist_batch = Categorical(logits=logits_batch)
                    actions_batch = dist_batch.sample()
                    log_probs_batch = dist_batch.log_prob(actions_batch)
                for i, agent in enumerate(env.agents):
                    action = actions_batch[i].item()
                    actions[agent.agent_id] = action
                    collected_data.append(
                        {
                            "agent_id": agent.agent_id,
                            "observation": obs_batch,
                            "action": actions_batch[i],
                            "log_prob": log_probs_batch[i],
                            "value": values_batch[i],
                        }
                    )
            else:
                obs_on_device = obs_batch.to(device)
                with torch.no_grad():
                    logits, value = policies(obs_on_device)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                agent = env.agents[0]
                actions[agent.agent_id] = action.item()
                collected_data.append(
                    {
                        "agent_id": agent.agent_id,
                        "observation": obs_batch,
                        "action": action,
                        "log_prob": log_prob,
                        "value": value,
                    }
                )
            next_obs, reward, done, info = env.step(actions)
            episode_step += 1
            experience = {
                "worker_id": worker_id,
                "episode_step": episode_step,
                "obs": current_obs,
                "next_obs": next_obs,
                "actions": actions,
                "collected_data": collected_data,
                "reward": reward,
                "done": done,
                "info": info,
            }
            data_queue.put(experience)
            if done:
                obs = env.reset()
                episode_step = 0
            else:
                obs = next_obs
    except KeyboardInterrupt:
        pass
    except Exception as e:
        data_queue.put(
            {
                "worker_id": worker_id,
                "error": str(e),
                "error_type": type(e).__name__,
            }
        )
        raise


if __name__ == "__main__":
    import argparse
    import time

    def test_worker():
        """Test the worker function locally with multiple workers."""
        parser = argparse.ArgumentParser(description="Test env_worker with multiple workers")
        parser.add_argument(
            "--num-steps", type=int, default=20, help="Total number of experiences to collect"
        )
        parser.add_argument(
            "--num-workers", type=int, default=4, help="Number of worker processes to spawn"
        )
        parser.add_argument(
            "--device", type=str, default="cpu", choices=["cpu", "mps"], help="Device to use"
        )
        args = parser.parse_args()

        # Create test configuration
        test_config = {
            "env": {
                "width": 30,
                "height": 20,
                "num_agents": 4,
                "total_seeds_per_episode": 500,
                "max_steps": 1000,
                "spawn_seed_every": 100,
                "seeds_per_spawn": 10,
                "policies": [],
            },
            "policy_state": None,
        }

        # Initialize policy and get state dict
        from simverse.envs.farmtila.config import FarmtilaConfig
        from simverse.envs.farmtila.env import FarmtilaEnv

        farm_config = FarmtilaConfig(**test_config["env"])
        env = FarmtilaEnv(farm_config)
        policy = SimplePolicy(env.observation_space, env.action_space)
        test_config["policy_state"] = policy.state_dict()

        # Create shared queue for all workers
        data_queue = mp.Queue()

        # Create and start multiple worker processes
        worker_processes = []
        print(f"🚀 Starting {args.num_workers} workers...")
        for worker_id in range(args.num_workers):
            worker_process = mp.Process(
                target=env_worker,
                args=(worker_id, test_config, data_queue, args.device),
            )
            worker_process.start()
            worker_processes.append(worker_process)
            print(f"  Worker {worker_id} started (PID: {worker_process.pid})")

        print(
            f"\n📊 Collecting {args.num_steps} total experiences from {args.num_workers} workers..."
        )

        experiences_collected = 0
        worker_stats = {i: {"count": 0, "episodes": 0} for i in range(args.num_workers)}
        start_time = time.time()

        try:
            while experiences_collected < args.num_steps:
                if not data_queue.empty():
                    experience = data_queue.get(timeout=1.0)

                    if "error" in experience:
                        worker_id = experience["worker_id"]
                        print(f"❌ Worker {worker_id} error: {experience['error']}")
                        continue

                    worker_id = experience["worker_id"]
                    experiences_collected += 1
                    worker_stats[worker_id]["count"] += 1

                    if experience["done"]:
                        worker_stats[worker_id]["episodes"] += 1

                    print(
                        f"  [{experiences_collected}/{args.num_steps}] Worker {worker_id}, "
                        f"Step {experience['episode_step']}, "
                        f"Done: {experience['done']}"
                    )
                else:
                    # Check if any workers are still alive
                    alive_workers = sum(1 for p in worker_processes if p.is_alive())
                    if alive_workers == 0:
                        print("⚠️  All workers have terminated")
                        break
                    time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted by user")
        finally:
            # Terminate all workers
            print(f"\n🛑 Terminating {len(worker_processes)} workers...")
            for i, worker_process in enumerate(worker_processes):
                if worker_process.is_alive():
                    worker_process.terminate()
                    worker_process.join(timeout=2.0)
                    if worker_process.is_alive():
                        worker_process.kill()
                        print(f"  Worker {i} force killed")
                    else:
                        print(f"  Worker {i} terminated")

            elapsed = time.time() - start_time

            # Print summary
            print(f"\n{'=' * 60}")
            print("📈 SUMMARY")
            print(f"{'=' * 60}")
            print(f"Total experiences collected: {experiences_collected}")
            print(f"Total time: {elapsed:.2f}s")
            rate = experiences_collected / max(elapsed, 1e-8)
            print(f"Overall rate: {rate:.2f} experiences/sec")
            print("\nPer-worker statistics:")
            for worker_id, stats in worker_stats.items():
                if stats["count"] > 0:
                    count = stats["count"]
                    episodes = stats["episodes"]
                    print(f"  Worker {worker_id}: {count} experiences, {episodes} episodes")
            print(f"{'=' * 60}")

    test_worker()
