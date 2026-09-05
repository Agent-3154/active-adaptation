"""Gripper semantic adaptation: joint/body indices + closedness helper.

Commands / rewards should use ``env.require_adaptation("gripper")`` instead of
re-resolving ``arm_joint[7,8]`` / ``gripper_(left|right)`` in every term.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from typing_extensions import override

from active_adaptation.envs.robots.adaptation import RobotAdaptation
from active_adaptation.envs.utils import find_bodies, find_joints

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from active_adaptation.envs.env_base import _EnvBase


class GripperAdaptation(RobotAdaptation):
    """Expose gripper joint/body indices and a normalized closedness signal.

    ``closedness()`` returns ``[num_envs, 1]`` in ``[0, 1]`` with ``0`` = fully
    open and ``1`` = fully closed (max |joint pos| / soft limit).
    """

    name = "gripper"

    def __init__(
        self,
        joint_names: str | Sequence[str] = "arm_joint[7,8]",
        body_names: str | Sequence[str] | None = "gripper_(left|right)",
    ) -> None:
        super().__init__()
        self.joint_names_cfg = joint_names
        self.body_names_cfg = body_names
        self.joint_ids: torch.Tensor | None = None
        self.joint_names: list[str] = []
        self.body_ids: torch.Tensor | None = None
        self.body_names: list[str] = []
        self.max_open: torch.Tensor | None = None

    @override
    def _initialize(self, env: "_EnvBase", *, robot: "Articulation") -> None:
        super()._initialize(env, robot=robot)

        joint_ids, joint_names = find_joints(robot, self.joint_names_cfg)
        if not joint_ids:
            raise ValueError(
                f"GripperAdaptation: no joints matched {self.joint_names_cfg!r}"
            )
        self.joint_ids = torch.as_tensor(
            joint_ids, device=self.device, dtype=torch.long
        )
        self.joint_names = list(joint_names)

        limits = robot.data.soft_joint_pos_limits[0, self.joint_ids]
        self.max_open = limits.abs().amax(dim=-1).max().clamp_min(1e-6)

        if self.body_names_cfg is None:
            self.body_ids = None
            self.body_names = []
            return

        body_ids, body_names = find_bodies(robot, self.body_names_cfg)
        if not body_ids:
            raise ValueError(
                f"GripperAdaptation: no bodies matched {self.body_names_cfg!r}"
            )
        self.body_ids = torch.as_tensor(body_ids, device=self.device, dtype=torch.long)
        self.body_names = list(body_names)

    def closedness(self) -> torch.Tensor:
        """Return gripper closedness in ``[0, 1]`` (0=open, 1=closed), shape ``[N, 1]``."""
        gripper_pos = self.robot.data.joint_pos[:, self.joint_ids]
        return (
            gripper_pos.abs().amax(dim=-1, keepdim=True) / self.max_open
        ).clamp(0.0, 1.0)

    def joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos[:, self.joint_ids]


__all__ = ["GripperAdaptation"]
