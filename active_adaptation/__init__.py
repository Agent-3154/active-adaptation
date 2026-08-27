import os
import sys
import json
import datetime
import builtins
import inspect
import importlib
import glob
import warp as wp

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

_RANK = int(os.getenv("RANK", os.getenv("LOCAL_RANK", "0")))
_LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
_WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
_MAIN_PROCESS = _RANK == 0
_ISAACLAB_EXCLUDED_EXTENSIONS = ("omni.warp.core",)


def is_main_process():
    return _MAIN_PROCESS


def is_distributed():
    return _WORLD_SIZE > 1


def get_local_rank():
    return _LOCAL_RANK


def get_rank():
    return _RANK


def get_world_size():
    return _WORLD_SIZE


def _append_kit_arg(existing: str, arg: str) -> str:
    existing = existing.strip()
    if not existing:
        return arg
    if arg in existing:
        return existing
    return f"{existing} {arg}"


def _apply_default_isaaclab_kit_args(app_config: dict) -> dict:
    kit_args = str(app_config.get("kit_args", "") or "")
    for index, extension in enumerate(_ISAACLAB_EXCLUDED_EXTENSIONS):
        kit_args = _append_kit_arg(
            kit_args,
            f"--/app/extensions/excluded/{index}={extension}",
        )
    app_config["kit_args"] = kit_args
    return app_config


def _expose_isaacsim_extension_modules() -> None:
    """Expose pip IsaacSim extension packages as importable Python modules."""

    try:
        import isaacsim
    except Exception:
        return

    isaacsim_root = Path(isaacsim.__file__).resolve().parent
    search_roots = [
        isaacsim_root / "exts",
        isaacsim_root / "extscache",
        isaacsim_root / "kit" / "extscore",
    ]
    for root in search_roots:
        for path in glob.glob(str(root / "*")):
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)

    for root in (isaacsim_root / "exts", isaacsim_root / "extscache"):
        for path in glob.glob(str(root / "*" / "isaacsim")):
            if os.path.isdir(path) and path not in isaacsim.__path__:
                isaacsim.__path__.append(path)


# Save original print function
_original_print = builtins.print


def _ranked_print(*args, **kwargs):
    """Print function with rank information prefix."""
    _original_print(
        f"[RANK {_RANK}/{_WORLD_SIZE} local={_LOCAL_RANK}]:", *args, **kwargs
    )


# Override builtins.print for global effect
if is_distributed():
    builtins.print = _ranked_print


CONFIG_PATH = Path(__file__).parent.parent / "cfg"
ASSET_PATH = Path(__file__).parent / "assets"
SCRIPT_PATH = Path(__file__).parent.parent / "scripts"
ROBOT_MODEL_DIR = CACHE_DIR / "aa-robot-models"


def set_backend(backend: str):
    global _BACKEND, _BACKEND_SET, _CALLED_AT
    if _BACKEND_SET:
        raise RuntimeError(
            f"set_backend() already called at {_CALLED_AT['filename']}:{_CALLED_AT['lineno']} in {_CALLED_AT['function']}"
        )
    if not backend in ("isaaclab", "mujoco", "mjlab"):
        raise ValueError(
            f"backend must be either 'isaaclab' or 'mujoco' or 'mjlab', got {backend}"
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

    if auto_rank:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.set_device(get_local_rank())
        except Exception:
            pass

    wp.init()

    # Store sys.argv to a local file
    if is_main_process():
        argv_file = CACHE_DIR / "command_history.json"
        if argv_file.exists():
            try:
                history = json.loads(argv_file.read_text())
            except Exception:
                history = []
        else:
            history = []
        entry = {"timestamp": datetime.datetime.now().isoformat(), "args": sys.argv}
        history.append(entry)
        argv_file.write_text(json.dumps(history, indent=2))

    set_backend(cfg.backend)
    if _BACKEND == "mjlab":
        cfg.device = "cuda"  # force to use GPU for mjlab
    elif _BACKEND == "mujoco":
        cfg.device = "cpu"  # force to use CPU for mujoco

    if auto_rank and (str(cfg.device) == "cuda"):
        cfg.device = f"cuda:{get_local_rank()}"

    if is_distributed():
        import torch
        import torch.distributed as dist

        if dist.is_available() and not dist.is_initialized():
            timeout_s = os.getenv("ACTIVE_ADAPTATION_DISTRIBUTED_TIMEOUT_S")
            timeout = None
            if timeout_s:
                timeout = datetime.timedelta(seconds=float(timeout_s))
            dist.init_process_group(
                backend="nccl",
                # world_size=get_world_size(),
                # rank=get_local_rank(),
                init_method="env://",
                **({"timeout": timeout} if timeout is not None else {}),
            )

    if get_backend() == "isaaclab":
        from isaaclab.app import AppLauncher

        app_config = OmegaConf.to_container(cfg.app, resolve=True)
        app_config = _apply_default_isaaclab_kit_args(app_config)
        AppLauncher(app_config, distributed=is_distributed(), device=cfg.device)
        _expose_isaacsim_extension_modules()

    import_environment_projects()

    return cfg
