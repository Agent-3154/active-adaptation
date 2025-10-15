import torch
import torch.nn as nn
from torch.utils._contextlib import _DecoratorContextManager


class VecNorm(nn.Module):
    """
    A more flexible version of EmpiricalNormalizer.
    This class allows you to normalize an observation of shape [*, C, H, W]
    with statistics of shape [C, 1, 1] instead of [C, H, W].

    Examples:

    Normalize an observation of shape [*, C, H, W] with statistics of shape [C, 1, 1]:
    >>> vecnorm = VecNorm(
        input_shape=(C, H, W),
        stats_shape=(C, 1, 1),
    )
    
    Args:
        input_shape: The shape of the input tensor.
        stat_shape: The shape of the statistics tensor.
        decay: The decay rate of the statistics.
    """
    
    FROZEN: bool = False

    def __init__(
        self,
        input_shape: torch.Size,
        stats_shape: torch.Size,
        decay: float=0.999,
    ):
        super().__init__()
        self.input_shape = torch.Size(input_shape)
        self.stats_shape = torch.Size(stats_shape)
        self.reduction_dims = []
        self.decay = decay

        _ = torch.broadcast_shapes(self.input_shape, self.stats_shape)

        count_factor = 1
        for dim in range(-1, -len(self.input_shape)-1, -1):
            if self.input_shape[dim] != self.stats_shape[dim]:
                self.reduction_dims.append(dim)
                count_factor *= self.input_shape[dim]

        self.register_buffer("sum", torch.zeros(self.stats_shape))
        self.register_buffer("ssq", torch.zeros(self.stats_shape))
        self.register_buffer("count", torch.tensor(1.0))
        # self.register_buffer("decay", torch.tensor(decay))
        self.register_buffer("count_factor", torch.tensor(count_factor))

        self.eps = torch.finfo(torch.float32).eps
    
    def __str__(self):
        return f"VecNorm(input_shape={self.input_shape}, stats_shape={self.stats_shape}, decay={self.decay}, reduction_dims={self.reduction_dims}, count_factor={self.count_factor})"
        
    def forward(self, input_vector: torch.Tensor):
        if not self.FROZEN:
            self._update(input_vector)
        mean, std = self._compute()
        return (input_vector - mean) / std
    
    def _update(self, input_vector: torch.Tensor):
        input_vector = input_vector.reshape(-1, *self.input_shape)
        sum_ = input_vector.mean(dim=self.reduction_dims, keepdim=True).sum(0)
        ssq_ = input_vector.square().mean(dim=self.reduction_dims, keepdim=True).sum(0)
        if self.decay < 1.0:
            weight = 1 - self.decay
            self.count.add_(input_vector.shape[0])
        else:
            weight = input_vector.shape[0] / self.count
            self.count.add_(input_vector.shape[0])
        self.sum.lerp_(end=sum_, weight=weight)
        self.ssq.lerp_(end=ssq_, weight=weight)
        
    def _compute(self):
        mean = self.sum / self.count
        std = (self.ssq / self.count - mean.pow(2)).clamp_min(self.eps).sqrt()
        return mean, std
    
    class freeze(_DecoratorContextManager):
        def __enter__(self):
            VecNorm.FROZEN = True
        
        def __exit__(self, exc_type, exc_value, traceback):
            VecNorm.FROZEN = False


if __name__ == "__main__":
    vecnorm = VecNorm(
        input_shape=(3, 4, 5),
        stats_shape=(3, 1, 1),
    )
    print(vecnorm)

    with VecNorm.freeze():
        for i in range(100):
            vecnorm(torch.randn(32, 10, 3, 4, 5))
    mean, std = vecnorm._compute()
    print(mean.squeeze(0), std.squeeze(0))

    for i in range(10):
        vecnorm(torch.randn(32, 10, 3, 4, 5))
    mean, std = vecnorm._compute()
    print(mean.squeeze(0), std.squeeze(0))

