"Abstraction for agent stats"

from abc import ABC, abstractmethod
from simverse.abstractor.agent import SimAgent

class AgentStats(ABC):
    @abstractmethod
    def __init__(self, agent: SimAgent):
        self.agent = agent
        self.stats = {}

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def update(self) -> None:
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        pass
