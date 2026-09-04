import os
import builtins
import inspect

from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from fractions import Fraction
from hydra.core.plugins import Plugins

from active_adaptation.project_loading.manifest import CACHE_DIR
from active_adaptation.project_loading.plugin import ActiveAdaptationSearchPathPlugin
from active_adaptation.project_loading.runtime import (
    import_environment_projects,
    resolve_wandb_defaults,
)

import active_adaptation.learning

OmegaConf.register_new_resolver("frac", lambda s: float(Fraction(s)))
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("rank", lambda: get_local_rank())
OmegaConf.register_new_resolver(
    "rank_select",
    lambda xs: xs[get_local_rank() % len(xs)],
)
Plugins.instance().register(ActiveAdaptationSearchPathPlugin)

_BACKEND = None
_BACKEND_SET = False
_CALLED_AT = None

_LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
_GLOBAL_RANK = int(os.getenv("RANK", str(_LOCAL_RANK)))
_WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
_MAIN_PROCESS = _GLOBAL_RANK == 0
_ISAACLAB_EXCLUDED_EXTENSIONS = ("omni.warp.core",)
# Set by :func:`isolate_local_cuda_device` when this process is pinned to one GPU.
_CVD_ISOLATED = False


def is_main_process():
    return _MAIN_PROCESS


def is_distributed():
    return _WORLD_SIZE > 1


def get_local_rank():
    return _LOCAL_RANK


def get_world_size():
    return _WORLD_SIZE


def get_local_cuda_index() -> int:
    """Process-local CUDA device index for Torch / Warp / DDP / NCCL.

    After :func:`isolate_local_cuda_device`, each rank sees a single GPU as
    ``cuda:0`` (required by Isaac USDRT, which only supports ``cuda:0``).
    Without isolation, this is ``LOCAL_RANK``.
    """
    if _CVD_ISOLATED:
        return 0
    return _LOCAL_RANK


def isolate_local_cuda_device() -> None:
    """Pin this process to one physical GPU before CUDA / Isaac init.

    ``launch_ddp`` sets ``CUDA_VISIBLE_DEVICES=0,1,…`` for the whole torchrun
    job, so rank 1 would otherwise use ``cuda:1``. Isaac's USDRT Fabric path
    (``UsdStage::SelectPrims``) only supports ``cuda:0`` and hangs on other
    devices — the failure mode for Lambert / MeshRegistry sensors under DDP.

    Remap visible devices so each rank's local ``cuda:0`` is a distinct GPU.
    Must run before any ``torch.cuda`` / SimulationApp initialization.
    """
    global _CVD_ISOLATED
    if _CVD_ISOLATED or not is_distributed():
        return
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd:
        return
    ids = [part.strip() for part in cvd.split(",") if part.strip()]
    if len(ids) <= 1:
        # Already a single device (or empty); treat as isolated for indexing.
        _CVD_ISOLATED = True
        return
    if _LOCAL_RANK >= len(ids):
        raise RuntimeError(
            f"LOCAL_RANK={_LOCAL_RANK} out of range for CUDA_VISIBLE_DEVICES={cvd!r}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ids[_LOCAL_RANK]
    _CVD_ISOLATED = True
    _original_print(
        f"[RANK {_LOCAL_RANK}/{_WORLD_SIZE}]: "
        f"isolated CUDA_VISIBLE_DEVICES -> {ids[_LOCAL_RANK]} "
        f"(process-local cuda:0)",
        flush=True,
    )


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
ROBOT_MODEL_DIR = CACHE_DIR / "aa-robot-models"


def set_backend(backend: str):
    global _BACKEND, _BACKEND_SET, _CALLED_AT
    if _BACKEND_SET:
        raise RuntimeError(
            f"set_backend() already called at {_CALLED_AT['filename']}:{_CALLED_AT['lineno']} in {_CALLED_AT['function']}"
        )
    if str(backend).lower() == "isaac": # alias for backward compatibility
        backend = "isaaclab"
    if not backend in ("isaaclab", "mujoco", "mjlab", "motrix"):
        raise ValueError(
            f"backend must be either 'isaaclab' or 'mujoco' or 'mjlab' or 'motrix', got {backend}"
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
    """Return None if the backend is not set."""
    return _BACKEND if _BACKEND_SET else None


def bind_local_rank_device() -> None:
    """Re-bind Torch (and Warp, if loaded) to this rank's CUDA device.

    Isaac Kit ``app.update`` / SimulationApp and Warp mesh init often reset
    ``torch.cuda.current_device()`` back to 0. NCCL then sees every rank on the
    same visible GPU and hangs at ``ncclCommInitRankConfig``. Call this after
    AppLauncher, after sensor/Warp setup, and immediately before collectives.
    """
    if not is_distributed():
        return
    import sys
    import torch
    import warp as wp

    if not torch.cuda.is_available():
        return
    wp.init()
    idx = get_local_cuda_index()
    torch.cuda.set_device(idx)
    wp.set_device(f"cuda:{idx}")


def _init_process_group() -> None:
    """Create the NCCL process group bound to this rank's CUDA device."""
    import torch
    import torch.distributed as dist

    if not dist.is_available() or dist.is_initialized():
        return
    bind_local_rank_device()
    kwargs: dict = {
        "backend": "nccl",
        "init_method": "env://",
    }
    # PyTorch 2.2+: pin the communicator to the local device so Kit/Warp
    # device resets cannot leave NCCL on cuda:0 for every rank.
    if torch.cuda.is_available():
        kwargs["device_id"] = torch.device(f"cuda:{get_local_cuda_index()}")
    try:
        dist.init_process_group(**kwargs)
    except TypeError:
        kwargs.pop("device_id", None)
        dist.init_process_group(**kwargs)


def init(cfg: DictConfig, auto_rank: bool):
    """Initialize the active adaptation framework.

    Args:
        cfg: The configuration dictionary.
        auto_rank: Whether to automatically modify `cfg.device` according to the local rank.
    """

    set_backend(cfg.backend)
    if _BACKEND == "mjlab":
        cfg.device = "cuda"  # force to use GPU for mjlab
    elif _BACKEND == "mujoco":
        cfg.device = "cpu"  # force to use CPU for mujoco
    elif _BACKEND == "motrix":
        pass # motrixsim env lives on CPU while policy training can be on GPU

    # Before any CUDA / Isaac init: one visible GPU per rank so USDRT (cuda:0
    # only) works under multi-GPU DDP with Lambert / MeshRegistry sensors.
    if is_distributed() and auto_rank:
        isolate_local_cuda_device()

    if auto_rank and str(cfg.device).startswith("cuda"):
        # Remap bare "cuda" / "cuda:0" etc. onto the process-local device index.
        cfg.device = f"cuda:{get_local_cuda_index()}"

    if is_distributed() and auto_rank:
        import torch

        # NCCL uses torch.cuda.current_device(). Without set_device, every rank
        # stays on visible cuda:0 → "Duplicate GPU detected".
        if torch.cuda.is_available():
            torch.cuda.set_device(get_local_cuda_index())

    if get_backend() == "isaaclab":
        from isaaclab.app import AppLauncher
        # viser and isaac have some conflicts, so we need to import viser here
        import viser

        app_config = OmegaConf.to_container(cfg.app, resolve=True)
        app_config = _apply_default_isaaclab_kit_args(app_config)
        # AppLauncher(distributed=True) sets active_gpu=LOCAL_RANK. After CVD
        # isolation the only valid local index is 0, so temporarily report
        # LOCAL_RANK=0 for Kit while keeping our module-level rank for NCCL.
        _saved_local_rank = os.environ.get("LOCAL_RANK")
        if _CVD_ISOLATED:
            os.environ["LOCAL_RANK"] = "0"
        try:
            AppLauncher(
                app_config,
                distributed=is_distributed(),
                device=cfg.device,
            )
        finally:
            if _saved_local_rank is not None:
                os.environ["LOCAL_RANK"] = _saved_local_rank
        # SimulationApp resets current CUDA device to 0; re-bind before NCCL.
        bind_local_rank_device()

    # Init NCCL *after* AppLauncher so the communicator is created on the
    # correct local device (Kit would otherwise undo a pre-AppLauncher bind).
    if is_distributed():
        _init_process_group()
        bind_local_rank_device()

    import active_adaptation.assets # register assets
    import active_adaptation.envs.sensors  # register sensors
    projects = import_environment_projects()

    default_wandb_api_key, default_wandb_project, default_wandb_entity = resolve_wandb_defaults(projects)
    if default_wandb_api_key is not None and not os.getenv("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = default_wandb_api_key
    if default_wandb_entity is not None and not os.getenv("WANDB_ENTITY"):
        os.environ["WANDB_ENTITY"] = default_wandb_entity
    if default_wandb_project is not None:
        wandb_cfg = cfg.get("wandb")
        if wandb_cfg is not None and "project" in wandb_cfg:
            wandb_cfg.project = default_wandb_project

    return cfg
