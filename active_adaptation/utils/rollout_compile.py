"""Configurable compilation of tensor-only rollout regions."""
from collections.abc import Callable
from types import FunctionType, MethodType

import torch


def compile_rollout_function(function: Callable, *, dynamic: bool = False) -> Callable:
    # Dynamo caches by code object. Give heterogeneous observation/reward groups
    # separate caches instead of spending one global recompile budget on all groups.
    if isinstance(function, MethodType):
        original = function.__func__
        isolated = FunctionType(
            original.__code__.replace(), original.__globals__, original.__name__,
            original.__defaults__, original.__closure__,
        )
        isolated.__kwdefaults__ = original.__kwdefaults__
        function = MethodType(isolated, function.__self__)
    # Use eager random kernels; fused execution need not preserve per-element draws.
    # Disable CUDA graphs to avoid output aliasing in rollout buffers.
    return torch.compile(
        function,
        fullgraph=True,
        dynamic=dynamic,
        options={"triton.cudagraphs": False, "fallback_random": True},
    )


def rollout_compile_enabled(cfg, region: str) -> bool:
    regions = cfg.get("rollout_compile", ("observations", "rewards", "nan_guard"))
    allowed = {"observations", "rewards", "nan_guard"}
    unknown = set(regions) - allowed
    if unknown:
        raise ValueError(f"Unknown rollout_compile regions: {sorted(unknown)}")
    return region in regions


def finite_row_masks(values: tuple[torch.Tensor, ...]):
    masks = tuple(~torch.isfinite(value).reshape(value.shape[0], -1).all(dim=1)
                  for value in values)
    invalid = torch.zeros_like(masks[0])
    for mask in masks:
        invalid = invalid | mask
    return invalid, masks


def sanitize_rows(values: tuple[torch.Tensor, ...], invalid: torch.Tensor):
    return tuple(torch.where(invalid.reshape(invalid.shape[0], *((1,) * (value.ndim - 1))),
                             torch.zeros_like(value), value) for value in values)
