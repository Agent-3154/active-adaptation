import torch
import torch.nn as nn
import torch.distributed as dist

from typing import Union
from torch.utils._contextlib import _DecoratorContextManager


class VecNorm(nn.Module):
    """
    A more flexible version of EmpiricalNormalizer.
    This class allows you to normalize an observation of shape [*, C, H, W]
    with statistics of shape [C, 1, 1] instead of [C, H, W].

    Decay behavior:
    - decay < 1.0:
      Uses exponentially decayed running moments. Recent batches are weighted
      more heavily than old batches, so the normalizer can track non-stationary
      observation distributions (e.g., curriculum, domain randomization, or
      changing policies during RL training). This is usually the safest default
      for online training.
    - decay = 1.0:
      Uses cumulative (non-decayed) moments over all seen samples. Every sample
      has equal weight in the long run, giving a stable global estimate when
      the observation distribution is approximately stationary. This is useful
      for fixed datasets or stable environments where you want long-horizon
      statistics without forgetting.

    Practical guidance:
    - Prefer decay < 1.0 for online RL where state distributions shift over time.
    - Prefer decay = 1.0 when data is stationary and you want exact cumulative
      normalization across all samples.

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
        decay: float=0.995,
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
        self.register_buffer("count", torch.tensor(0.0))
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
        return self._normalize(input_vector)

    def _normalize(self, input_vector: torch.Tensor):
        mean, std = self._compute()
        return (input_vector - mean) / std

    def denormalize(self, input_vector: torch.Tensor):
        mean, std = self._compute()
        return input_vector * std + mean

    def _update(self, input_vector: torch.Tensor):
        input_vector = input_vector.reshape(-1, *self.input_shape).float()
        if len(self.reduction_dims):
            # note that `tensor.mean(())` is not what we want
            sum_ = input_vector.mean(dim=self.reduction_dims, keepdim=True)
            ssq_ = input_vector.square().mean(dim=self.reduction_dims, keepdim=True)
        else:
            sum_ = input_vector
            ssq_ = input_vector.square()
        # Keep running-stat updates in buffer dtype (float32 by default).
        # This avoids in-place dtype mismatches for fp16/bf16 inputs and is
        # numerically safer for long-horizon accumulation.
        sum_ = sum_.to(self.sum.dtype)
        ssq_ = ssq_.to(self.ssq.dtype)
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
            denom = self.count.clamp_min(1.0)
            mean = self.sum / denom
            var = (self.ssq / denom - mean.pow(2)).clamp_min(self.eps)
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

        with torch.no_grad():
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
                    # Non-EMA: buffers store means; aggregate using per-rank counts.
                    # Use float64 accumulators to reduce overflow/precision loss.
                    mean_buf = self.sum.to(dtype=torch.float64)
                    sqmean_buf = self.ssq.to(dtype=torch.float64)
                    count_buf = self.count.to(dtype=torch.float64)
                    weighted_mean = mean_buf * count_buf
                    weighted_sqmean = sqmean_buf * count_buf
                    dist.all_reduce(weighted_mean, op=dist.ReduceOp.SUM)
                    dist.all_reduce(weighted_sqmean, op=dist.ReduceOp.SUM)
                    dist.all_reduce(count_buf, op=dist.ReduceOp.SUM)

                    count_global = count_buf.clamp_min(1.0)
                    mean_global = (weighted_mean / count_global).to(dtype=self.sum.dtype)
                    sqmean_global = (weighted_sqmean / count_global).to(dtype=self.ssq.dtype)
                    self.sum.copy_(mean_global)
                    self.ssq.copy_(sqmean_global)
                    self.count.copy_(count_buf.to(dtype=self.count.dtype))
            else:
                raise ValueError(f"Invalid mode: {mode}")
    

    class freeze(_DecoratorContextManager):
        def __enter__(self):
            self.prev_state = VecNorm.FROZEN
            VecNorm.FROZEN = True
        
        def __exit__(self, exc_type, exc_value, traceback):
            VecNorm.FROZEN = self.prev_state


class VecNormRMS(VecNorm):
    """
    RMS-only variant of `VecNorm`.

    Difference from `VecNorm`:
    - `VecNorm` performs mean-centered z-score normalization:
      `(x - mean) / std`, where `std = sqrt(E[x^2] - E[x]^2)`.
    - `VecNormRMS` performs scale-only RMS normalization:
      `x / rms`, where `rms = sqrt(E[x^2])`.

    Because it does not subtract the running mean, `VecNormRMS` preserves
    input sign for non-zero values and is often preferred when the sign carries
    semantic meaning (e.g., directional signals).
    """

    def __repr__(self):
        return (
            f"VecNormRMS(input_shape={self.input_shape}, stats_shape={self.stats_shape}, "
            f"decay={self.decay}, reduction_dims={self.reduction_dims}, "
            f"count_factor={self.count_factor})"
        )

    def _compute_rms(self):
        if self.decay < 1.0:
            denom = self.count.clamp_min(1.0)
            mean_sq = self.ssq / denom
        else:
            mean_sq = self.ssq
        return mean_sq.clamp_min(self.eps).sqrt()

    def _normalize(self, input_vector: torch.Tensor):
        rms = self._compute_rms()
        return input_vector / rms

    def denormalize(self, input_vector: torch.Tensor):
        rms = self._compute_rms()
        return input_vector * rms


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
