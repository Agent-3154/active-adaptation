"""Entity behaviors: composable entity-attached logic (not Articulation subclasses).

A behavior owns semantic indices / caches and optional physics hooks. Assets
(robots **or** scene objects) declare a list on
:class:`~active_adaptation.assets.asset_cfg.AssetSpec`; the env binds them after
the entity exists and calls lifecycle methods explicitly.

Examples: underwater hydrodynamics, gripper closedness, object grasp sampling.

Lookup keys on ``env.behaviors``:

- Robot behaviors: ``behavior.name`` (e.g. ``"gripper"``).
- Object behaviors: ``{scene_object_name}.{behavior.name}`` (e.g. ``"object.grasp"``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDictBase

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


class EntityBehavior:
    """Asset-attached behavior (composition over inheritance).

    Subclasses must set :attr:`name`. Construct with config only; bind via
    :meth:`_initialize` after the scene entity exists.

    The bound handle is :attr:`asset` (robot articulation or object entity).
    :attr:`robot` is a compatibility alias for the same reference.
    """

    name: str = ""

    def __init__(self) -> None:
        if not self.name:
            raise TypeError(f"{type(self).__name__} must define a non-empty class attr `name`")
        self.env: _EnvBase | None = None
        self.asset: Any | None = None
        self._initialized = False

    def _initialize(
        self,
        env: "_EnvBase",
        *,
        asset: Any | None = None,
        robot: Any | None = None,
    ) -> None:
        """Bind to a scene entity.

        Prefer ``asset=``. ``robot=`` remains accepted as a deprecated alias.
        """
        if asset is None:
            asset = robot
        if asset is None:
            raise TypeError(f"{type(self).__name__}._initialize requires asset=")
        self.env = env
        self.asset = asset
        self._initialized = True

    @property
    def robot(self) -> Any | None:
        """Compatibility alias for :attr:`asset` (robot-bound behaviors)."""
        return self.asset

    @robot.setter
    def robot(self, value: Any) -> None:
        self.asset = value

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
            return self.asset.device
        except AttributeError:
            return self.asset.data.root_link_pos_w.device

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


__all__ = ["EntityBehavior"]
