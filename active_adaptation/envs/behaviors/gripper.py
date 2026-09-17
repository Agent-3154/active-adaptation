"""Gripper semantic behavior: EEF + finger indices and closedness.

Commands / rewards should use ``env.require_behavior("gripper")`` instead of
re-resolving grasp / finger names in every term.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from typing_extensions import override

from active_adaptation.envs.behaviors.behavior import EntityBehavior
from active_adaptation.envs.utils import find_bodies, find_joints
from active_adaptation.utils.math import normalize, quat_rotate, quat_rotate_inverse

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from active_adaptation.envs.env_base import _EnvBase


class GripperBehavior(EntityBehavior):
    """Expose EEF body, finger joints/bodies, and a normalized closedness signal.

    **Frame:** unless an asset documents otherwise, the EEF body ``+X`` is
    **forward / approach** (see ``grasp_pose.EEF_FORWARD_B`` / ``eef_forward_w``).

    Assets must spawn gripper joints **open** (see ``INIT_JOINT_POS`` on
    ``a2_manipulator``) so ``finger_seg`` is well-defined at ``_initialize``.
    Closed rest is ``q ≈ 0``; open / init is ``|q|`` toward the soft limit
    (A2 Piper / similar). Then:

    - ``openness()`` → ``[N, 1]`` in ``[0, 1]`` with ``0`` = closed rest, ``1`` = open
    - ``closedness()`` → ``1 - openness()``
    """

    name = "gripper"

    def __init__(
        self,
        eef_body_name: str = "grasp_point",
        joint_names: str | Sequence[str] = "arm_joint[7,8]",
        body_names: str | Sequence[str] | None = "gripper_(left|right)",
    ) -> None:
        super().__init__()
        self.eef_body_name_cfg = eef_body_name
        self.joint_names_cfg = joint_names
        self.body_names_cfg = body_names
        self.eef_body_id: int = -1
        self.eef_body_name: str = ""
        self.joint_ids: torch.Tensor | None = None
        self.joint_names: list[str] = []
        self.body_ids: torch.Tensor | None = None
        self.body_names: list[str] = []
        self.max_open: torch.Tensor | None = None

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
        
        finger_left, finger_right = self.finger_pos_w().unbind(dim=1)
        finger_seg_w = finger_right - finger_left
        if (finger_seg_w.norm(dim=-1, keepdim=True) < 1e-2).any():
            raise ValueError(f"GripperBehavior: finger segment is too short")
        self._finger_seg = quat_rotate_inverse(self.eef_quat_w, normalize(finger_seg_w))

    @property
    def eef_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.eef_body_id]

    @property
    def eef_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.eef_body_id]
    
    def finger_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_ids]
    
    def finger_seg_normalized_w(self) -> torch.Tensor:
        """Finger segment in the world frame, normalized."""
        return quat_rotate(self.eef_quat_w, self._finger_seg)

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
