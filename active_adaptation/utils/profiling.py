import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils._contextlib import _DecoratorContextManager
from typing import Any, List, Dict, Optional

PROFILE_EXPORT_ENV = "AA_PROFILE_EXPORT"
PROFILE_PRINT_ENV = "AA_PROFILE_PRINT"
PROFILE_PRINT_EVERY_ENV = "AA_PROFILE_PRINT_EVERY"
PROFILE_JSONL_FILENAME = "profiling.jsonl"
PROFILE_SYNC_TIMERS = os.environ.get("AA_PROFILE_SYNC_TIMERS", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def profile_export_enabled() -> bool:
    return os.environ.get(PROFILE_EXPORT_ENV, "1").lower() in {"1", "true", "yes"}


def profile_print_enabled() -> bool:
    return os.environ.get(PROFILE_PRINT_ENV, "1").lower() in {"1", "true", "yes"}


def profile_print_every() -> int:
    raw = os.environ.get(PROFILE_PRINT_EVERY_ENV, "1")
    try:
        return max(1, int(raw))
    except ValueError:
        return 1

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
    def _roots() -> List["ScopedTimer"]:
        roots = ScopedTimer._root_nodes or [
            t for t in ScopedTimer._instances.values() if t.parent is None
        ]
        if not roots:
            roots = sorted(ScopedTimer._instances.values(), key=lambda t: t.name)
        return roots

    @staticmethod
    def _timer_record(
        node: "ScopedTimer",
        *,
        path: str,
        depth: int,
        total_time: float,
    ) -> dict[str, Any]:
        avg_ms = node.time / node.count * 1000 if node.count else 0.0
        pct = node.time / total_time * 100 if total_time else 0.0
        return {
            "path": path,
            "depth": depth,
            "count": node.count,
            "total_s": node.time,
            "avg_ms": avg_ms,
            "pct": pct,
        }

    @staticmethod
    def _collect_nodes(
        node: "ScopedTimer",
        *,
        path_prefix: str,
        depth: int,
        max_depth: int,
        total_time: float,
        out: list[dict[str, Any]],
    ) -> None:
        path = f"{path_prefix}/{node.name}" if path_prefix else node.name
        out.append(
            ScopedTimer._timer_record(
                node,
                path=path,
                depth=depth,
                total_time=total_time,
            )
        )
        if max_depth > 0 and depth + 1 >= max_depth:
            return
        for child in node.children:
            ScopedTimer._collect_nodes(
                child,
                path_prefix=path,
                depth=depth + 1,
                max_depth=max_depth,
                total_time=total_time,
                out=out,
            )

    @staticmethod
    def collect_summary(depth: int = -1) -> dict[str, Any]:
        """Return a structured timing summary without clearing timers.

        ``depth`` is the maximum tree depth to include (levels 0 .. depth-1). Use
        ``depth <= 0`` for no limit (full tree).
        """
        if not ScopedTimer._instances:
            return {"roots": [], "total_s": 0.0, "timers": []}

        roots = ScopedTimer._roots()
        total_time = sum(r.time for r in roots)
        max_depth = depth if depth > 0 else -1
        timers: list[dict[str, Any]] = []
        for root in roots:
            ScopedTimer._collect_nodes(
                root,
                path_prefix="",
                depth=0,
                max_depth=max_depth,
                total_time=total_time,
                out=timers,
            )
        return {
            "roots": [r.name for r in roots],
            "total_s": total_time,
            "timers": sorted(timers, key=lambda row: row["pct"], reverse=True),
        }

    @staticmethod
    def clear_summary() -> None:
        for timer in ScopedTimer._instances.values():
            timer.time = 0.0
            timer.count = 0

    @staticmethod
    def export_summary(clear: bool = True, depth: int = -1) -> dict[str, Any]:
        """Export timing summary as a machine-readable dict (full tree by default)."""
        summary = ScopedTimer.collect_summary(depth=depth)
        if clear:
            ScopedTimer.clear_summary()
        return summary

    @staticmethod
    def append_profiling_jsonl(path: Path | str, record: dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

    @staticmethod
    def _print_node(
        node: "ScopedTimer",
        depth: int,
        max_depth: int,
        total_time: float,
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
        for child in node.children:
            ScopedTimer._print_node(child, depth + 1, max_depth, total_time, color)

    @staticmethod
    def print_summary(clear: bool = True, depth: int = 3, use_color: Optional[bool] = None):
        """Print timing summary for all timers.

        ``depth`` is the maximum tree depth to print (levels 0 .. depth-1). Use
        ``depth <= 0`` for no limit (print the full tree).
        """
        summary = ScopedTimer.collect_summary(depth=depth)
        if not summary["timers"]:
            print("No timers recorded.")
            if clear:
                ScopedTimer.clear_summary()
            return

        color = sys.stdout.isatty() if use_color is None else use_color
        max_depth = depth if depth > 0 else -1
        roots = ScopedTimer._roots()
        total_time = summary["total_s"]

        print("\n" + "=" * _SEP_W)
        print(
            f"{'Timer Name':<{_NAME_W}} {'Count':>8} {'Total (s)':>10} "
            f"{'Avg (ms)':>10} {'%':>8}"
        )
        print("=" * _SEP_W)
        for root in roots:
            ScopedTimer._print_node(root, 0, max_depth, total_time, color)
        print("=" * _SEP_W + "\n")
        if clear:
            ScopedTimer.clear_summary()
