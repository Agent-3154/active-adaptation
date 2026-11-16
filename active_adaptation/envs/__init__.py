import importlib

from .env_base import _EnvBase
from . import terrain

_BACKEND_ENV_EXPORTS = {
    "IsaacBackendEnv": "active_adaptation.envs.backends.isaaclab",
    "MjlabBackendEnv": "active_adaptation.envs.backends.mjlab",
    "MotrixBackendEnv": "active_adaptation.envs.backends.motrix",
}


def __getattr__(name: str):
    module_name = _BACKEND_ENV_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value

__all__ = [
    "_EnvBase",
    "IsaacBackendEnv",
    "MjlabBackendEnv",
    "MotrixBackendEnv",
    "terrain",
]
