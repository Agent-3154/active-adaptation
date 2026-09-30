"""Gripper semantic behavior: EEF, open axis, and closedness.

Commands / rewards should use ``env.require_behavior("gripper")`` instead of
re-resolving grasp / finger names in every term.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from typing_extensions import override

from active_adaptation.envs.behaviors.behavior import EntityBehavior
from active_adaptation.envs.utils import find_bodies, find_joints
from active_adaptation.utils.math import quat_rotate

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from active_adaptation.envs.env_base import _EnvBase


class GripperBehavior(EntityBehavior):
    """Expose EEF body, finger joints/bodies, and a normalized closedness signal.

    **Frame:** unless an asset documents otherwise, the EEF body ``+X`` is
    **forward / approach** (see ``grasp_pose.EEF_FORWARD_B`` / ``eef_forward_w``).
    ``open_direction`` is the jaw opening axis in that same EEF frame
    (A2 Piper: ``+Y``; Spot arm: ``+Z``). ``open_direction_w`` rotates it with
    the EEF.

    Closed rest is ``q ≈ 0``; open / init is ``|q|`` toward the soft limit.
    Then:

    - ``openness()`` → ``[N, 1]`` in ``[0, 1]`` with ``0`` = closed rest, ``1`` = open
    - ``closedness()`` → ``1 - openness()``
    """

    name = "gripper"

    def __init__(
        self,
        eef_body_name: str = "grasp_point",
        joint_names: str | Sequence[str] = "arm_joint[7,8]",
        body_names: str | Sequence[str] | None = "gripper_(left|right)",
        open_direction: Sequence[float] = (0.0, 1.0, 0.0),
    ) -> None:
        super().__init__()
        direction = tuple(float(x) for x in open_direction)
        if len(direction) != 3:
            raise ValueError(
                f"GripperBehavior: open_direction must have 3 components, got {open_direction!r}"
            )
        self.eef_body_name_cfg = eef_body_name
        self.joint_names_cfg = joint_names
        self.body_names_cfg = body_names
        self.open_direction_cfg = direction
        self.eef_body_id: int = -1
        self.eef_body_name: str = ""
        self.joint_ids: torch.Tensor | None = None
        self.joint_names: list[str] = []
        self.body_ids: torch.Tensor | None = None
        self.body_names: list[str] = []
        self.max_open: torch.Tensor | None = None
        self._open_direction_b: torch.Tensor | None = None

    @override
    def _initialize(self, env: "_EnvBase", *, asset: "Articulation", robot: "Articulation | None" = None) -> None:
        super()._initialize(env, asset=asset, robot=robot)

        eef_ids, eef_names = find_bodies(self.asset, self.eef_body_name_cfg)
        if len(eef_ids) != 1:
            raise ValueError(
                f"GripperBehavior: expected one EEF body for "
                f"{self.eef_body_name_cfg!r}, got {eef_names}"
            )
        self.eef_body_id = int(eef_ids[0])
        self.eef_body_name = eef_names[0]

        joint_ids, joint_names = find_joints(self.asset, self.joint_names_cfg)
        if not joint_ids:
            raise ValueError(
                f"GripperBehavior: no joints matched {self.joint_names_cfg!r}"
            )
        self.joint_ids = torch.as_tensor(
            joint_ids, device=self.device, dtype=torch.long
        )
        self.joint_names = list(joint_names)

        limits = self.asset.data.soft_joint_pos_limits[0, self.joint_ids]
        self.max_open = limits.abs().amax(dim=-1).max().clamp_min(1e-6)

        direction = torch.tensor(
            self.open_direction_cfg, device=self.device, dtype=torch.float32
        )
        norm = direction.norm()
        if norm < 1e-6:
            raise ValueError(f"GripperBehavior: open_direction is zero")
        self._open_direction_b = (direction / norm).view(1, 3)

        if self.body_names_cfg is None:
            self.body_ids = None
            self.body_names = []
            return

        body_ids, body_names = find_bodies(self.asset, self.body_names_cfg)
        if not body_ids:
            raise ValueError(
                f"GripperBehavior: no bodies matched {self.body_names_cfg!r}"
            )
        self.body_ids = torch.as_tensor(body_ids, device=self.device, dtype=torch.long)
        self.body_names = list(body_names)

    @property
    def eef_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.eef_body_id]

    @property
    def eef_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.eef_body_id]

    def finger_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.body_ids]

    def open_direction_w(self) -> torch.Tensor:
        """Unit open axis in the world frame, shape ``[N, 3]``."""
        return quat_rotate(self.eef_quat_w, self._open_direction_b)

    def joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos[:, self.joint_ids]

    def openness(self) -> torch.Tensor:
        """Finger opening in ``[0, 1]`` (0=closed rest, 1=at soft limit), ``[N, 1]``."""
        return (
            self.joint_pos().abs().amax(dim=-1, keepdim=True) / self.max_open
        ).clamp(0.0, 1.0)

    def closedness(self) -> torch.Tensor:
        """Gripper closedness in ``[0, 1]`` (0=open, 1=closed), shape ``[N, 1]``."""
        return 1.0 - self.openness()


__all__ = ["GripperBehavior"]
