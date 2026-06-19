from __future__ import annotations

import torch

from typing import TYPE_CHECKING
from typing_extensions import override

from .base import ActionV2


if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


class WriteRootState(ActionV2):
    def __init__(self):
        super().__init__()

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.action_dim = 13
        self.target_root_pose = None
        self.target_root_velocity = None

    def process_action(self, action: torch.Tensor):
        self.target_root_pose = action[:, :7]
        self.target_root_pose[:, :3] += self.env.scene.env_origins
        self.target_root_velocity = action[:, 7:]

    @override
    def apply_action(self, substep: int):
        if self.target_root_pose is None:
            return
        self.asset.write_root_pose_to_sim(self.target_root_pose)
        self.asset.write_root_velocity_to_sim(self.target_root_velocity)


class WriteJointPosition(ActionV2):
    def __init__(self):
        super().__init__()

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.action_dim = self.asset.data.default_joint_pos.shape[-1]
        self.target_joint_pos = None

    def process_action(self, action: torch.Tensor):
        self.target_joint_pos = action

    @override
    def apply_action(self, substep: int):
        if self.target_joint_pos is None:
            return
        self.asset.set_joint_position_target(self.target_joint_pos)
        self.asset.write_joint_position_to_sim(self.target_joint_pos)
        self.asset.write_joint_velocity_to_sim(torch.zeros_like(self.target_joint_pos))


__all__ = ["WriteRootState", "WriteJointPosition"]
