from pathlib import Path

from simverse.core.trainer import Trainer
from simverse.training.checkpoints import Checkpointer


class DummyPolicy:
    def __init__(self):
        self._state = {"weight": 1}

    def state_dict(self):
        return dict(self._state)

    def load_state_dict(self, state_dict):
        self._state = dict(state_dict)


class DummyAgent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.policy = DummyPolicy()


class DummyEnv:
    def __init__(self):
        self.config = {"name": "dummy"}
        self.steps = 7
        self.agents = [DummyAgent(0), DummyAgent(1)]


class DummyTrainer(Trainer):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def train(self, *args, **kwargs) -> None:
        return None


def test_checkpointer_reuses_same_run_directory_within_a_run(tmp_path: Path) -> None:
    checkpointer = Checkpointer(DummyEnv())

    first_path = checkpointer.save(str(tmp_path / "checkpoints" / "episode_0.pth"))
    second_path = checkpointer.save(str(tmp_path / "checkpoints" / "episode_1.pth"))

    assert first_path.parent == second_path.parent
    assert first_path.parent.name == checkpointer.run_id
    assert first_path.exists()
    assert second_path.exists()


def test_checkpointer_creates_unique_run_directory_per_instance(tmp_path: Path) -> None:
    first_checkpointer = Checkpointer(DummyEnv())
    second_checkpointer = Checkpointer(DummyEnv())

    first_path = first_checkpointer.save(str(tmp_path / "checkpoints" / "episode_0.pth"))
    second_path = second_checkpointer.save(str(tmp_path / "checkpoints" / "episode_0.pth"))

    assert first_checkpointer.run_id != second_checkpointer.run_id
    assert first_path.parent != second_path.parent


def test_trainer_reuses_one_checkpointer_for_all_saves(tmp_path: Path) -> None:
    trainer = DummyTrainer(DummyEnv())

    trainer.save_checkpoint(str(tmp_path / "checkpoints" / "episode_0.pth"))
    trainer.save_checkpoint(str(tmp_path / "checkpoints" / "episode_1.pth"))

    run_directories = sorted((tmp_path / "checkpoints").iterdir())
    assert len(run_directories) == 1
    assert sorted(path.name for path in run_directories[0].iterdir()) == [
        "episode_0.pth",
        "episode_1.pth",
    ]
