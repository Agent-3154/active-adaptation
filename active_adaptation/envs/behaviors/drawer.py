"""Drawer articulation behavior: slide joint accessors (no locks).

Attach via ``AssetSpec(behaviors=(DrawerBehavior(), ...))``. Lookup with
``env.require_behavior("drawer.drawer")`` when the YAML object key is ``drawer``.

Supports **multi-drawer** cabinets (``drawer_{i}_joint``). Handles are welded
bars — there is no handle joint and no lock/unlock logic.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import torch
from typing_extensions import override

from active_adaptation.envs.behaviors.behavior import EntityBehavior
from active_adaptation.envs.utils import find_joints

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase

_DRAWER_JOINT_RE = re.compile(r"^drawer_(\d+)_joint$")


class DrawerBehavior(EntityBehavior):
    """Expose multi-drawer slide joints (unlocked / free-sliding)."""

    name = "drawer"

    def __init__(
        self,
        drawer_joint_name: str = r"drawer_\d+_joint",
    ) -> None:
        super().__init__()
        self.drawer_joint_name_cfg = drawer_joint_name
        self.num_drawers: int = 0
        self.drawer_joint_ids: list[int] = []

    @override
    def _initialize(
        self,
        env: "_EnvBase",
        *,
        asset: Any | None = None,
        robot: Any | None = None,
    ) -> None:
        super()._initialize(env, asset=asset, robot=robot)

        drawer_ids, drawer_names = find_joints(self.asset, self.drawer_joint_name_cfg)
        if len(drawer_ids) == 0:
            raise ValueError(
                f"DrawerBehavior: no drawer joints matching "
                f"{self.drawer_joint_name_cfg!r}"
            )

        def _index(name: str) -> int:
            m = _DRAWER_JOINT_RE.match(name)
            if m is None:
                raise ValueError(f"Unexpected joint name {name!r}")
            return int(m.group(1))

        drawer_pairs = sorted(
            zip(drawer_ids, drawer_names),
            key=lambda p: _index(p[1]),
        )
        self.drawer_joint_ids = [int(i) for i, _ in drawer_pairs]
        self.num_drawers = len(self.drawer_joint_ids)

    @property
    def drawer_joint_pos(self) -> torch.Tensor:
        """``[N, K]`` slide positions."""
        return self.asset.data.joint_pos[:, self.drawer_joint_ids]


__all__ = ["DrawerBehavior"]
