from typing import TypeVar, cast

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class _DDPWithAttr(DDP):
    """Internal :class:`DDP` subclass that forwards unknown attribute lookups to the wrapped module.

    Prefer :func:`wrap_ddp` over instantiating this directly; the factory
    preserves the wrapped module's static type for the IDE / type checker.
    """

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            module = self.__dict__.get("_modules", {}).get("module")
            if module is not None and hasattr(module, name):
                return getattr(module, name)
            raise


_ModuleT = TypeVar("_ModuleT", bound=nn.Module)


def wrap_ddp(module: _ModuleT, **kwargs) -> _ModuleT:
    """Wrap ``module`` in DDP while keeping the static type of ``module``.

    The returned value is a :class:`DistributedDataParallel` at runtime (with
    attribute forwarding to the wrapped module), but the static type checker /
    IDE treats it as the original module's type so all of its attributes and
    methods stay visible. Use :func:`isinstance` against :class:`DDP` (or
    :func:`unwrap_ddp`) when you specifically need to address the wrapper
    (e.g. :meth:`DDP.no_sync`).
    """
    return cast(_ModuleT, _DDPWithAttr(module, **kwargs))


def unwrap_ddp(module: _ModuleT) -> _ModuleT:
    """Return the underlying module if ``module`` is a :class:`DDP`, else ``module``."""
    inner = module.module if isinstance(module, DDP) else module
    return cast(_ModuleT, inner)



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


def check_gradients(module_or_param: torch.nn.Module | nn.Parameter):
    """
    Check if gradients on different GPUs are the same.

    Return the maximum absolute difference between gradients on different GPUs.
    """
    if isinstance(module_or_param, nn.Parameter):
        return _max_abs_diff_across_ranks(module_or_param.grad)
    m = unwrap_ddp(module_or_param)
    max_diff = 0.0
    for p in m.parameters():
        if p.grad is not None:
            max_diff = max(max_diff, _max_abs_diff_across_ranks(p.grad))
    return max_diff


def check_parameters(module_or_param: torch.nn.Module | nn.Parameter):
    """
    Check if parameters on different GPUs are the same.

    Return the maximum absolute difference between parameters on different GPUs.
    """
    if isinstance(module_or_param, nn.Parameter):
        return _max_abs_diff_across_ranks(module_or_param.data)
    m = unwrap_ddp(module_or_param)
    max_diff = 0.0
    for p in m.parameters():
        max_diff = max(max_diff, _max_abs_diff_across_ranks(p.data))
    return max_diff


__all__ = ["wrap_ddp", "unwrap_ddp", "check_gradients", "check_parameters"]

