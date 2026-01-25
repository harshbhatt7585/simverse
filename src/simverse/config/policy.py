from dataclasses import dataclass
from simverse.abstractor.policy import Policy

@dataclass
class PolicySpec:
    name: str
    model: Policy

    