import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, DDP) else module


def _max_abs_diff_across_ranks(t: torch.Tensor) -> float:
    """Max over elements of (max_i t_i - min_i t_i) after gathering flat tensors from all ranks."""
    if not dist.is_available() or not dist.is_initialized():
        return 0.0
    world_size = dist.get_world_size()
    if world_size <= 1:
        return 0.0
    flat = t.detach().contiguous().view(-1).float()
    chunks = [torch.empty_like(flat) for _ in range(world_size)]
    dist.all_gather(chunks, flat)
    stacked = torch.stack(chunks, dim=0)
    span = (stacked.max(dim=0).values - stacked.min(dim=0).values).max()
    return float(span.item())


def check_gradients(module: torch.nn.Module):
    """
    Check if gradients on different GPUs are the same.

    Return the maximum absolute difference between gradients on different GPUs.
    """
    m = _unwrap(module)
    max_diff = 0.0
    for p in m.parameters():
        if p.grad is not None:
            max_diff = max(max_diff, _max_abs_diff_across_ranks(p.grad))
    return max_diff


def check_parameters(module: torch.nn.Module):
    """
    Check if parameters on different GPUs are the same.

    Return the maximum absolute difference between parameters on different GPUs.
    """
    m = _unwrap(module)
    max_diff = 0.0
    for p in m.parameters():
        max_diff = max(max_diff, _max_abs_diff_across_ranks(p.data))
    return max_diff
