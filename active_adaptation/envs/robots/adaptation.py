"""Robot adaptations: composable asset-attached behavior (not articulation subclasses).

An adaptation owns semantic indices / caches and optional physics hooks. Assets
declare a list on :class:`~active_adaptation.assets.asset_cfg.AssetSpec`; the
env binds them after the robot exists and calls lifecycle methods explicitly.

Examples: underwater hydrodynamics, gripper closedness helpers.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDictBase

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from active_adaptation.envs.env_base import _EnvBase


class RobotAdaptation:
    """Asset-attached robot behavior (composition, not inheritance of Articulation).

    Subclasses must set :attr:`name` (unique key on ``env.adaptations``).
    Construct with config only; bind via :meth:`_initialize` after the scene
    robot exists.
    """

    name: str = ""

    def __init__(self) -> None:
        if not self.name:
            raise TypeError(f"{type(self).__name__} must define a non-empty class attr `name`")
        self.env: _EnvBase | None = None
        self.robot: Articulation | None = None
        self._initialized = False

    def _initialize(self, env: "_EnvBase", *, robot: "Articulation") -> None:
        self.env = env
        self.robot = robot
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def num_envs(self) -> int:
        if not self._initialized:
            raise RuntimeError(f"{type(self).__name__} is not initialized")
        return self.env.num_envs

    @property
    def device(self) -> torch.device:
        if not self._initialized:
            raise RuntimeError(f"{type(self).__name__} is not initialized")
        try:
            return self.robot.device
        except AttributeError:
            return self.robot.data.root_link_pos_w.device

    # --- lifecycle (env calls these explicitly; base methods are no-ops) ---

    def startup(self) -> None:
        pass

    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase) -> None:
        pass

    def pre_step(self, substep: int) -> None:
        pass

    def post_step(self, substep: int) -> None:
        pass

    def update(self, tensordict: TensorDictBase | None = None) -> None:
        pass

    def debug_draw(self) -> None:
        pass

    def edit_spec(self, scene_config: Any) -> None:
        """Optional: mutate backend scene cfg before scene construction."""
        pass


__all__ = ["RobotAdaptation"]
