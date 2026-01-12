from __future__ import annotations

import json
import os
import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import active_adaptation as aa


class CommandHistory:
    """Persistent launcher command history manager."""

    def __init__(self, path: Path | None = None, max_entries: int = 100) -> None:
        self.path: Path = path if path is not None else (aa.SCRIPT_PATH / "commands.json")
        self.max_entries: int = max_entries
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return []

    def save(self, entries: List[Dict[str, Any]]) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entries[-self.max_entries :], indent=2))
        tmp.replace(self.path)

    def add(self, entry: Dict[str, Any]) -> None:
        entries = self.load()
        entries.append(entry)
        self.save(entries)

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        entries = self.load()
        return list(reversed(entries[-limit:]))

    @staticmethod
    def make_entry(
        task: str,
        algo: str,
        use_ddp: bool,
        gpus: List[str],
        cmd: List[str],
        pid: int,
        cwd: str,
        timestamp: str | None = None,
    ) -> Dict[str, Any]:
        return {
            "timestamp": timestamp or (datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"),
            "task": task,
            "algo": algo,
            "use_ddp": use_ddp,
            "gpus": gpus,
            "cmd": cmd,
            "pid": pid,
            "cwd": cwd,
        }

    def to_dropdown_choices(self, limit: int = 50) -> List[Tuple[str, str]]:
        """Return list of (label, value_json) tuples for UI dropdown."""
        choices: List[Tuple[str, str]] = []
        for e in self.recent(limit=limit):
            ts = e.get("timestamp", "")
            task = e.get("task", "")
            algo = e.get("algo", "")
            mode = "DDP" if e.get("use_ddp") else "single"
            gpus = ",".join(e.get("gpus", []) or [])
            label = f"[{ts}] {task} / {algo}  ({mode} {gpus})"
            choices.append((label, json.dumps(e)))
        return choices

    @staticmethod
    def selection_to_controls(value_json: str) -> Tuple[str, str, bool, List[str]]:
        """Map a selected history entry back to controls: task, algo, use_ddp, gpus."""
        try:
            e = json.loads(value_json or "{}")
        except Exception:
            e = {}
        return (
            e.get("task", ""),
            e.get("algo", ""),
            bool(e.get("use_ddp", False)),
            list(e.get("gpus", []) or []),
        )


