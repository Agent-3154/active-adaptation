from . import ppo
from . import offpolicy
from . import imitation

# Register algo=from_checkpoint for play / rollout / eval.
import active_adaptation.utils.checkpoint_cfg  # noqa: F401

__all__ = [
    "ppo",
    "offpolicy",
    "imitation",
]
