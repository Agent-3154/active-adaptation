import torch
import torch.nn as nn
import math
import torch.distributed as dist
import active_adaptation

from typing import Union
from torch.utils._contextlib import _DecoratorContextManager
from torchrl.envs.transforms import VecNorm


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
        input_shape: Union[torch.Size, tuple, int],
        stats_shape: Union[torch.Size, tuple, int]=None,
        decay: float=0.999,
    ):
        super().__init__()
        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        if stats_shape is None:
            stats_shape = input_shape
        elif isinstance(stats_shape, int):
            stats_shape = (stats_shape,)
        self.input_shape = torch.Size(input_shape)
        self.stats_shape = torch.Size(stats_shape)
        self.decay = decay

        _ = torch.broadcast_shapes(self.input_shape, self.stats_shape)

        count_factor = 1
        reduction_dims = []
        for dim in range(-1, -len(self.input_shape)-1, -1):
            if self.input_shape[dim] != self.stats_shape[dim]:
                reduction_dims.append(dim)
                count_factor *= self.input_shape[dim]
        self.reduction_dims = tuple(reduction_dims)

        self.register_buffer("sum", torch.zeros(self.stats_shape))
        self.register_buffer("ssq", torch.zeros(self.stats_shape))
        self.register_buffer("count", torch.tensor(1.0))
        # self.register_buffer("decay", torch.tensor(decay))
        self.register_buffer("count_factor", torch.tensor(count_factor))
        self.sum: torch.Tensor
        self.ssq: torch.Tensor
        self.count: torch.Tensor
        self.count_factor: torch.Tensor

        self.eps = 1e-5 # torch.finfo(torch.float32).eps
    
    def __repr__(self):
        return f"VecNorm(input_shape={self.input_shape}, stats_shape={self.stats_shape}, decay={self.decay}, reduction_dims={self.reduction_dims}, count_factor={self.count_factor})"
        
    def forward(self, input_vector: torch.Tensor):
        if not self.FROZEN:
            self._update(input_vector)
        mean, std = self._compute()
        return (input_vector - mean) / std
    
    def _update(self, input_vector: torch.Tensor):
        input_vector = input_vector.reshape(-1, *self.input_shape)
        if len(self.reduction_dims):
            # note that `tensor.mean(())` is not what we want
            sum_ = input_vector.mean(dim=self.reduction_dims, keepdim=True)
            ssq_ = input_vector.square().mean(dim=self.reduction_dims, keepdim=True)
        else:
            sum_ = input_vector
            ssq_ = input_vector.square()
        if self.decay < 1.0:
            self.count.mul_(self.decay).add_(input_vector.shape[0])
            self.sum.mul_(self.decay).add_(sum_.sum(0))
            self.ssq.mul_(self.decay).add_(ssq_.sum(0))
        else:
            self.count.add_(input_vector.shape[0])
            weight = input_vector.shape[0] / self.count
            self.sum.lerp_(end=sum_.mean(0), weight=weight)
            self.ssq.lerp_(end=ssq_.mean(0), weight=weight)
        
    def _compute(self):
        if self.decay < 1.0:
            mean = self.sum / self.count
            var = (self.ssq / self.count - mean.pow(2)).clamp_min(self.eps)
        else:
            mean = self.sum
            var = (self.ssq - mean.pow(2)).clamp_min(self.eps)
        std = var.sqrt()
        return mean, std
    
    def synchronize(self, mode: str="broadcast"):
        """
        Synchronize the statistics across all ranks.
        Args:
            mode: The mode to synchronize the statistics.
                - "broadcast": Use rank 0's stats to update local stats.
                - "aggregate": Aggregate the statistics across all ranks.
        """
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Distributed training is not initialized")

        if mode == "broadcast":
            # Make all ranks identical to rank 0 by broadcasting buffers
            dist.broadcast(self.sum, src=0)
            dist.broadcast(self.ssq, src=0)
            dist.broadcast(self.count, src=0)
        elif mode == "aggregate":
            # Aggregate raw moments across ranks
            if self.decay < 1.0:
                # EMA case: buffers store weighted sums (numerators) and effective counts
                dist.all_reduce(self.sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(self.ssq, op=dist.ReduceOp.SUM)
                dist.all_reduce(self.count, op=dist.ReduceOp.SUM)
            else:
                # Non-EMA: buffers store means; with equal per-rank counts, average across ranks.
                world_size = dist.get_world_size()
                # Use float64 accumulators to reduce risk of overflow/precision loss
                mean_buf = self.sum.to(dtype=torch.float64)
                m2_buf = self.ssq.to(dtype=torch.float64)
                dist.all_reduce(mean_buf, op=dist.ReduceOp.SUM)
                dist.all_reduce(m2_buf, op=dist.ReduceOp.SUM)
                mean_global = (mean_buf / world_size).to(dtype=self.sum.dtype)
                m2_global = (m2_buf / world_size).to(dtype=self.ssq.dtype)
                self.sum.copy_(mean_global)
                self.ssq.copy_(m2_global)
        else:
            raise ValueError(f"Invalid mode: {mode}")
    

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

    for i in range(100):
        vecnorm(torch.randn(32, 10, 3, 4, 5))
    mean, std = vecnorm._compute()
    print(mean.squeeze(0), std.squeeze(0))

    vecnorm = VecNorm(
        input_shape=(4,),
        stats_shape=(4,),
        decay=1.0
    )
    print(vecnorm)

    with VecNorm.freeze():
        for i in range(100):
            vecnorm(torch.randn(4096, 4) * torch.tensor([1, -2, 3, -4]))
    mean, std = vecnorm._compute()
    print(mean.squeeze(0), std.squeeze(0))

    for i in range(500):
        vecnorm(torch.randn(4096, 4) * torch.tensor([1, -2, 3, -4]))
    mean, std = vecnorm._compute()
    print(mean.squeeze(0), std.squeeze(0))

