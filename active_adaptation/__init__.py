import os
import sys
import json
import datetime
import builtins
import inspect
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from fractions import Fraction
from hydra.core.plugins import Plugins

from active_adaptation.project_loading.manifest import CACHE_DIR
from active_adaptation.project_loading.plugin import ActiveAdaptationSearchPathPlugin
from active_adaptation.project_loading.runtime import import_environment_projects

import active_adaptation.learning

OmegaConf.register_new_resolver("frac", lambda s: float(Fraction(s)))
OmegaConf.register_new_resolver("eval", eval)
Plugins.instance().register(ActiveAdaptationSearchPathPlugin)

_BACKEND = None
_BACKEND_SET = False
_CALLED_AT = None

_LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
_WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
_MAIN_PROCESS = _LOCAL_RANK == 0


def is_main_process():
    return _MAIN_PROCESS


def is_distributed():
    return _WORLD_SIZE > 1


def get_local_rank():
    return _LOCAL_RANK


def get_world_size():
    return _WORLD_SIZE


# Save original print function
_original_print = builtins.print


def _ranked_print(*args, **kwargs):
    """Print function with rank information prefix."""
    _original_print(f"[RANK {_LOCAL_RANK}/{_WORLD_SIZE}]:", *args, **kwargs)


# Override builtins.print for global effect
if is_distributed():
    builtins.print = _ranked_print


CONFIG_PATH = Path(__file__).parent.parent / "cfg"
ASSET_PATH = Path(__file__).parent / "assets"
SCRIPT_PATH = Path(__file__).parent.parent / "scripts"


def set_backend(backend: str):
    global _BACKEND, _BACKEND_SET, _CALLED_AT
    if _BACKEND_SET:
        raise RuntimeError(
            f"set_backend() already called at {_CALLED_AT['filename']}:{_CALLED_AT['lineno']} in {_CALLED_AT['function']}"
        )
    if not backend in ("isaac", "mujoco", "mjlab"):
        raise ValueError(
            f"backend must be either 'isaac' or 'mujoco' or 'mjlab', got {backend}"
        )
    # Record the call site
    stack = inspect.stack()
    caller = stack[1]
    _BACKEND = backend
    _BACKEND_SET = True
    _CALLED_AT = {
        "filename": caller.filename,
        "lineno": caller.lineno,
        "function": caller.function,
        "code_context": caller.code_context[0].strip() if caller.code_context else None,
    }


def get_backend():
    if not _BACKEND_SET:
        raise RuntimeError("set_backend() must be called before get_backend()")
    return _BACKEND


def init(cfg: DictConfig, auto_rank: bool):
    """Initialize the active adaptation framework.

    Args:
        cfg: The configuration dictionary.
        auto_rank: Whether to automatically modify `cfg.device` according to the local rank.
    """

    # Store sys.argv to a local file
    if is_main_process():
        argv_file = CACHE_DIR / "command_history.json"
        if argv_file.exists():
            history = json.loads(argv_file.read_text())
        else:
            history = []
        entry = {"timestamp": datetime.datetime.now().isoformat(), "args": sys.argv}
        history.append(entry)
        argv_file.write_text(json.dumps(history, indent=2))

    set_backend(cfg["backend"])

    if auto_rank and (str(cfg.device) == "cuda"):
        cfg.device = f"cuda:{get_local_rank()}"

    if is_distributed():
        import torch
        import torch.distributed as dist

        if dist.is_available() and not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                # world_size=get_world_size(),
                # rank=get_local_rank(),
                # device_id=torch.device(cfg.device)
                init_method="env://",
            )

    if get_backend() == "isaac":
        from isaaclab.app import AppLauncher

        app_config = OmegaConf.to_container(cfg.app)
        AppLauncher(app_config, distributed=is_distributed(), device=cfg.device)

    import_environment_projects()

    return cfg
