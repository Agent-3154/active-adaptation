import os
import active_adaptation.learning
import builtins
import importlib.metadata
import importlib.util
from pathlib import Path

_BACKEND = "isaac"

_LOCAL_RANK = os.getenv("LOCAL_RANK", "0")
_LOCAL_RANK = int(_LOCAL_RANK)
_WORLD_SIZE = os.getenv("WORLD_SIZE", "1")
_WORLD_SIZE = int(_WORLD_SIZE)
_MAIN_PROCESS = _LOCAL_RANK == 0

_OMP_NUM_THREADS = os.getenv("OMP_NUM_THREADS", "1")
_OMP_NUM_THREADS = int(_OMP_NUM_THREADS)


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

if is_distributed() and _OMP_NUM_THREADS <= 1:
    raise ValueError("Please set OMP_NUM_THREADS to a value greater than 1 when using distributed training.")


ASSET_PATH = os.path.join(os.path.dirname(__file__), "assets")

def set_backend(backend: str):
    if not backend in ("isaac", "mujoco"):
        raise ValueError(f"backend must be either 'isaac' or 'mujoco', got {backend}")
    global _BACKEND
    _BACKEND = backend


def get_backend():
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


def import_projects():
    for entry_point in _PROJECT_ENTRY_POINTS:
        print(f"Importing project {entry_point.name}")
        entry_point.load()


def import_algorithms():
    for entry_point in _LEARNING_ENTRY_POINTS:
        print(f"Importing learning {entry_point.name}")
        entry_point.load()


