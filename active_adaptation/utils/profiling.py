import sys
import time
import torch
from torch.utils._contextlib import _DecoratorContextManager
from typing import List, Dict, Optional

_NAME_W = 36
_SEP_W = _NAME_W + 8 + 10 + 10 + 8 + 4
_LEVEL_ANSI = ("\033[36m", "\033[32m", "\033[33m", "\033[34m", "\033[35m", "\033[90m")
_RESET = "\033[0m"


class ScopedTimer(_DecoratorContextManager):
    """A context manager for timing code blocks with singleton pattern.

    Each timer name creates a singleton instance that accumulates timing data
    across multiple uses.

    Usage:
    >>> with ScopedTimer("step"):
    ...     time.sleep(1)
    >>> with ScopedTimer("step"):  # Reuses the same timer
    ...     time.sleep(1)
    >>> print(timer.last_time)
    1.0
    >>> ScopedTimer.print_summary(clear=True)
    """

    _instances: Dict[str, "ScopedTimer"] = {}
    _stack: List["ScopedTimer"] = []
    _root_nodes: List["ScopedTimer"] = []
    children: List["ScopedTimer"] = []

    def __new__(cls, name: str, sync: bool = False):
        if name not in cls._instances:
            instance = super().__new__(cls)
            instance.name = name
            instance.sync = sync
            instance.time = 0.0
            instance.count = 0
            instance.children = []
            instance.parent = None
            cls._instances[name] = instance
        return cls._instances[name]

    def __init__(self, name: str, sync: bool = False):
        self.sync = sync

    def clone(self):
        """Required by ``_DecoratorContextManager`` for decorator usage."""
        return self

    def __enter__(self):
        if self.parent is None:
            parent = ScopedTimer._stack[-1] if ScopedTimer._stack else None
            if parent is None:
                if self not in ScopedTimer._root_nodes:
                    ScopedTimer._root_nodes.append(self)
            else:
                self.parent = parent
                parent.children.append(self)
                if self in ScopedTimer._root_nodes:
                    ScopedTimer._root_nodes.remove(self)
        ScopedTimer._stack.append(self)
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.sync:
            torch.cuda.synchronize()
        self.last_time = time.perf_counter() - self.start
        self.time += self.last_time
        self.count += 1
        ScopedTimer._stack.pop()

    @staticmethod
    def _print_node(
        node: "ScopedTimer",
        depth: int,
        max_depth: int,
        total_time: float,
        clear: bool,
        color: bool,
    ) -> None:
        if max_depth > 0 and depth >= max_depth:
            return
        name = f"{'  ' * depth}{node.name}"
        avg_ms = node.time / node.count * 1000 if node.count else 0.0
        pct = node.time / total_time * 100 if total_time else 0.0
        if color:
            c = _LEVEL_ANSI[depth % len(_LEVEL_ANSI)]
            pad = max(_NAME_W - len(name), 0)
            print(
                f"{c}{name}{_RESET}{' ' * pad}"
                f" {c}{node.count:>8}{_RESET}"
                f" {c}{node.time:>10.4f}{_RESET}"
                f" {c}{avg_ms:>10.2f}{_RESET}"
                f" {c}{pct:>7.1f}%{_RESET}"
            )
        else:
            print(
                f"{name:<{_NAME_W}} {node.count:>8} {node.time:>10.4f} "
                f"{avg_ms:>10.2f} {pct:>7.1f}%"
            )
        if clear:
            node.time = 0.0
            node.count = 0
        for child in node.children:
            ScopedTimer._print_node(child, depth + 1, max_depth, total_time, clear, color)

    @staticmethod
    def print_summary(clear: bool = True, depth: int = 3, use_color: Optional[bool] = None):
        """Print timing summary for all timers.

        ``depth`` is the maximum tree depth to print (levels 0 .. depth-1). Use
        ``depth <= 0`` for no limit (print the full tree).
        """
        if not ScopedTimer._instances:
            print("No timers recorded.")
            return

        roots = ScopedTimer._root_nodes or [
            t for t in ScopedTimer._instances.values() if t.parent is None
        ]
        if not roots:
            roots = sorted(ScopedTimer._instances.values(), key=lambda t: t.name)

        total_time = sum(r.time for r in roots)
        color = sys.stdout.isatty() if use_color is None else use_color
        max_depth = depth if depth > 0 else -1

        print("\n" + "=" * _SEP_W)
        print(
            f"{'Timer Name':<{_NAME_W}} {'Count':>8} {'Total (s)':>10} "
            f"{'Avg (ms)':>10} {'%':>8}"
        )
        print("=" * _SEP_W)
        for root in roots:
            ScopedTimer._print_node(root, 0, max_depth, total_time, clear, color)
        print("=" * _SEP_W + "\n")
