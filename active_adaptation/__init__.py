import os
import sys
import json
import datetime
import active_adaptation.learning
import builtins
import importlib.metadata
import importlib.util
import inspect
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from fractions import Fraction

OmegaConf.register_new_resolver("frac", lambda s: float(Fraction(s)))
OmegaConf.register_new_resolver("eval", eval)

_BACKEND = None
_BACKEND_SET = False
_CALLED_AT = None

_LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
_RANK = int(os.getenv("RANK", _LOCAL_RANK))
_WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
_MAIN_PROCESS = _RANK == 0

_OMP_NUM_THREADS = os.getenv("OMP_NUM_THREADS", "1")
_OMP_NUM_THREADS = int(_OMP_NUM_THREADS)


def is_main_process():
    return _MAIN_PROCESS

def is_distributed():
    return _WORLD_SIZE > 1

def get_rank():
    return _RANK

def get_local_rank():
    return _LOCAL_RANK

def get_world_size():
    return _WORLD_SIZE

# Save original print function
_original_print = builtins.print

def _ranked_print(*args, **kwargs):
    """Print function with rank information prefix."""
    _original_print(f"[RANK {_RANK}/{_WORLD_SIZE} LOCAL {_LOCAL_RANK}]:", *args, **kwargs)

# Override builtins.print for global effect
if is_distributed():
    builtins.print = _ranked_print

if is_distributed() and _OMP_NUM_THREADS <= 1:
    raise ValueError("Please set OMP_NUM_THREADS to a value greater than 1 when using distributed training.")

CONFIG_PATH = Path(__file__).parent.parent / "cfg"
ASSET_PATH = Path(__file__).parent / "assets"
SCRIPT_PATH = Path(__file__).parent.parent / "scripts"


def set_backend(backend: str):
    global _BACKEND, _BACKEND_SET, _CALLED_AT
    if _BACKEND_SET:
        raise RuntimeError(f"set_backend() already called at {_CALLED_AT['filename']}:{_CALLED_AT['lineno']} in {_CALLED_AT['function']}")
    if not backend in ("isaac", "mujoco", "mjlab"):
        raise ValueError(f"backend must be either 'isaac' or 'mujoco' or 'mjlab', got {backend}")
    # Record the call site
    stack = inspect.stack()
    caller = stack[1]
    _BACKEND = backend
    _BACKEND_SET = True
    _CALLED_AT = {
        'filename': caller.filename,
        'lineno': caller.lineno,
        'function': caller.function,
        'code_context': caller.code_context[0].strip() if caller.code_context else None,
    }


def get_backend():
    if not _BACKEND_SET:
        raise RuntimeError("set_backend() must be called before get_backend()")
    return _BACKEND


_CONFIG_SEARCH_PATHS = []
_PROJECT_ENTRY_POINTS = []
_LEARNING_ENTRY_POINTS = []


for entry_point in importlib.metadata.entry_points(group="active_adaptation.projects"):
    # get the module path
    spec = importlib.util.find_spec(entry_point.value)
    _CONFIG_SEARCH_PATHS.append(str(Path(spec.origin).parent.absolute()))
    _PROJECT_ENTRY_POINTS.append(entry_point)


for entry_point in importlib.metadata.entry_points(group="active_adaptation.learning"):
    _LEARNING_ENTRY_POINTS.append(entry_point)


from hydra_plugins.aa_searchpath_plugin.aa_searchpath_plugin import ActiveAdaptationSearchPathPlugin


def _import_projects():
    for entry_point in _PROJECT_ENTRY_POINTS:
        print(f"Importing project {entry_point.name}")
        entry_point.load()


def import_algorithms():
    for entry_point in _LEARNING_ENTRY_POINTS:
        print(f"Importing learning {entry_point.name}")
        entry_point.load()


def init(
    cfg: DictConfig,
    auto_rank: bool,
    import_projects: bool = True,
):
    """Initialize the active adaptation framework.
    
    Args:
        cfg: The configuration dictionary.
        auto_rank: Whether to automatically modify `cfg.device` according to the local rank.
        import_projects: Whether to import the projects.
    """

    # Store sys.argv to a local file
    if is_main_process():
        argv_file = SCRIPT_PATH / "command_history.json"
        argv_file.parent.mkdir(parents=True, exist_ok=True)
        
        if argv_file.exists():
            history = json.loads(argv_file.read_text())
        else:
            history = []
        
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "args": sys.argv
        }
        history.append(entry)
        argv_file.write_text(json.dumps(history, indent=2))
    
    set_backend(cfg.get("backend", "isaac"))
    
    if auto_rank and str(cfg.device).startswith("cuda"):
        cfg.device = f"cuda:{get_local_rank()}"

    if get_backend() == "isaac":
        from isaaclab.app import AppLauncher
        app_config = OmegaConf.to_container(cfg.app)
        AppLauncher(app_config, distributed=is_distributed(), device=cfg.device)
    if import_projects:
        _import_projects()
    
    import_algorithms()

    return cfg
    