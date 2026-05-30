import torch
import torch.nn as nn
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Iterator, Tuple
from torch.utils.hooks import RemovableHandle

ACTIVATIONS = (nn.ReLU, nn.GELU, nn.Mish, nn.SiLU)


class DormancyTracker:
    """
    Track post-activation magnitude statistics to estimate dormant-unit ratios.

    Dormancy is computed following Sokar et al. (ICML 2023):
    https://arxiv.org/abs/2302.12902

    Typical usage:
        model = MyModel()
        tracker = DormancyTracker(model)

        # Track a specific forward section.
        with tracker.track():
            model(batch)

        # Or wrap a callable to track every call.
        tracked_forward = tracker.wrap(model)
        tracked_forward(batch)

    """

    def __init__(self, module: nn.Module) -> None:
        self._track_depth = 0
        self._handles: list[RemovableHandle] = []
        # module_name -> (sum |h| per unit, rows seen), accumulated as float32.
        self._stats: dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        def make_hook(module_name: str):
            @torch.no_grad()
            def hook(mod: nn.Module, _inp, out: torch.Tensor) -> None:
                if self._track_depth == 0:
                    return

                if not torch.is_tensor(out):
                    return

                x = out.detach()
                # Treat the last dimension as units and flatten all leading dims into rows.
                if x.ndim == 0:
                    return
                if x.ndim == 1:
                    x = x.unsqueeze(0)
                else:
                    x = x.reshape(-1, x.shape[-1])
                x = x.float()
                b, d = x.shape

                if module_name not in self._stats:
                    self._stats[module_name] = (
                        torch.zeros(d, dtype=torch.float32, device=x.device),
                        torch.zeros((), dtype=torch.float32, device=x.device),
                    )
                sum_abs, n_seen = self._stats[module_name]
                if sum_abs.shape[0] != d or sum_abs.device != x.device:
                    # If shape/device changes, restart this module's accumulation.
                    sum_abs, n_seen = (
                        torch.zeros(d, dtype=torch.float32, device=x.device),
                        torch.zeros((), dtype=torch.float32, device=x.device),
                    )
                    self._stats[module_name] = (sum_abs, n_seen)
                sum_abs.add_(x.abs().sum(dim=0))
                n_seen.add_(b)

            return hook

        for name, child in module.named_modules():
            if isinstance(child, ACTIVATIONS):
                handle = child.register_forward_hook(make_hook(name))
                self._handles.append(handle)

    def compute_dormancy(self, tau: float = 0.01) -> dict[str, float]:
        """
        Return dormant-unit fraction per tracked activation module.

        A unit is counted as dormant when its normalized mean absolute activation
        is <= ``tau``.
        """
        if tau < 0:
            raise ValueError(f"tau must be non-negative, got {tau}")
        eps = 1e-12
        out: dict[str, float] = {}
        for name, (sum_abs, n_seen) in self._stats.items():
            n = float(n_seen.item())
            if n <= 0:
                continue
            mean_abs = sum_abs / n
            layer_mean = mean_abs.mean()
            s = mean_abs / (layer_mean + eps)
            out[name] = float((s <= tau).float().mean().item())
        return out

    @contextmanager
    def track(self) -> Iterator[None]:
        """Context manager that enables statistic collection within its scope."""
        self._track_depth += 1
        try:
            yield
        finally:
            self._track_depth -= 1

    def reset(self) -> None:
        """Clear all accumulated activation statistics."""
        self._stats.clear()

    def close(self) -> None:
        """Remove forward hooks registered by this tracker."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def wrap(self, func: Callable) -> Callable:
        """Wrap ``func`` so every invocation runs under ``track()``."""
        @wraps(func)
        def wrapped(*args, **kwargs):
            with self.track():
                return func(*args, **kwargs)

        wrapped.__wrapped__ = func
        return wrapped

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
