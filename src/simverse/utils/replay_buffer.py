from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Experience:
    observation: Dict 
    action: List[int]
    log_prob: List[float]
    value: List[float]
    reward: List[float]
    done: List[bool]
    info: List[Dict]


class ReplayBuffer:
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
    
    def add(self, experience: tuple) -> None:
        pass

    def sample(self, batch_size: int) -> List[Experience]:
        pass