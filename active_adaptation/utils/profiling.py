import time
import torch
from torch.utils._contextlib import _DecoratorContextManager
from typing import List, Dict, Tuple


class ScopedTimer(_DecoratorContextManager):
    """A context manager for timing code blocks with singleton pattern.

    Each timer name creates a singleton instance that accumulates timing data
    across multiple uses.

    Usage:
    >>> with ScopedTimer("step"):
    ...     time.sleep(1)
    >>> with ScopedTimer("step"):  # Reuses the same timer
    ...     time.sleep(1)
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
        # Update sync flag if it's different
        self.sync = sync

    def clone(self):
        """Required by ``_DecoratorContextManager`` for decorator usage.

        For our singleton timer, cloning just returns ``self``, which is
        sufficient since timing state is accumulated per named timer.
        """
        return self

    def _detach(self):
        """Detach this timer from its current parent or the root list."""
        if self.parent is None:
            if self in ScopedTimer._root_nodes:
                ScopedTimer._root_nodes.remove(self)
            return

        if self in self.parent.children:
            self.parent.children.remove(self)
        self.parent = None

    def _attach(self, parent: "ScopedTimer | None"):
        """Attach this timer to a new parent or the root list."""
        self.parent = parent
        if parent is None:
            if self not in ScopedTimer._root_nodes:
                ScopedTimer._root_nodes.append(self)
            return

        if self not in parent.children:
            parent.children.append(self)

    def __enter__(self):
        # Update hierarchical parent/roots based on current stack at each use,
        # so timers appear under the scopes where they are most recently used.
        parent = ScopedTimer._stack[-1] if ScopedTimer._stack else None

        if parent is None:
            attached = self in ScopedTimer._root_nodes
        else:
            attached = self in parent.children

        # Re-parent if needed so the summary tree reflects current usage.
        if self.parent is not parent or not attached:
            self._detach()
            self._attach(parent)

        ScopedTimer._stack.append(self)
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.sync:
            torch.cuda.synchronize()
        self.end = time.perf_counter()
        self.last_time = self.end - self.start
        self.time += self.last_time
        self.count += 1
        ScopedTimer._stack.pop()

    @staticmethod
    def _roots_for_summary() -> Tuple[List["ScopedTimer"], bool]:
        """Timers with no parent are roots; fall back if _root_nodes desynced.

        Returns (roots, flat) where ``flat`` means print without recursing
        children (used when no node has parent=None).
        """
        if ScopedTimer._root_nodes:
            return list(ScopedTimer._root_nodes), False
        inferred = [t for t in ScopedTimer._instances.values() if t.parent is None]
        if inferred:
            return inferred, False
        # No explicit roots (e.g. inconsistent tree): show every timer once, flat.
        return sorted(ScopedTimer._instances.values(), key=lambda t: t.name), True

    @staticmethod
    def print_summary(clear: bool = True, depth: int = 3):
        """Print timing summary for all timers.

        ``depth`` is the maximum tree depth to print (levels 0 .. depth-1). Use
        ``depth <= 0`` for no limit (print the full tree).
        """
        if not ScopedTimer._instances:
            print("No timers recorded.")
            return

        roots, flat = ScopedTimer._roots_for_summary()
        total_time = sum(r.time for r in roots)
        print("\n" + "=" * 70)
        print(f"{'Timer Name':<30} {'Count':>8} {'Total (s)':>10} {'Avg (ms)':>10} {'%':>8}")
        print("=" * 70)

        if flat:
            for node in roots:
                avg_ms = (node.time / node.count * 1000) if node.count > 0 else 0
                pct = (node.time / total_time * 100) if total_time > 0 else 0.0
                print(
                    f"{node.name:<30} {node.count:>8} {node.time:>10.4f} {avg_ms:>10.2f} {pct:>7.1f}%"
                )
                if clear:
                    node.time = 0
                    node.count = 0
        else:
            eff_depth = depth if depth > 0 else -1
            for root in roots:
                root.print_recursive(
                    root, 0, clear=clear, max_depth=eff_depth, total_time=total_time
                )

        print("=" * 70 + "\n")

    def print_recursive(self, node: "ScopedTimer", depth: int = 0, clear: bool = True, max_depth: int = -1, total_time: float = 0.0):
        """Recursively print timer nodes in DFS order."""
        # max_depth <= 0 means no limit; otherwise print while depth < max_depth.
        if max_depth > 0 and depth >= max_depth:
            return
        indent = "  " * depth
        avg_ms = (node.time / node.count * 1000) if node.count > 0 else 0
        pct = (node.time / total_time * 100) if total_time > 0 else 0.0
        print(
            f"{indent}{node.name:<30} {node.count:>8} {node.time:>10.4f} {avg_ms:>10.2f} {pct:>7.1f}%"
        )
        if clear:
            node.time = 0
            node.count = 0

        for child in node.children:
            self.print_recursive(child, depth + 1, clear=clear, max_depth=max_depth, total_time=total_time)

