from simverse.losses.loss import Loss
import torch

class L2(Loss):
    def __init__(
        self, 
        epsilon: float = 1e-8,
        **kwargs,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon

    def __call__(self, target: torch.Tensor, prediction: torch.Tensor) -> float:
        return torch.mean(torch.square(target - prediction)) + self.epsilon
        