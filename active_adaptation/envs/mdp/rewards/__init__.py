# ruff: noqa: F401

import importlib
import os

from .base import Reward


_OPTIONAL_MODULES = {"isaaclab", "omni"}
_dir_path = os.path.dirname(os.path.realpath(__file__))

for _file in os.listdir(_dir_path):
    if _file.startswith("_") or _file in {"__init__.py", "base.py", "common.py"}:
        continue
    if not _file.endswith(".py"):
        continue

    try:
        importlib.import_module(f".{_file[:-3]}", __package__)
    except ModuleNotFoundError as exc:
        if exc.name in _OPTIONAL_MODULES:
            continue
        raise
    except ImportError:
        continue


__all__ = ["Reward"]
