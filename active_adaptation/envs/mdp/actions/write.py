from __future__ import annotations

import torch

from typing_extensions import override

from .base import Action


class WriteRootState(Action):
    def __init__(self, env):
        super().__init__(env)
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


class WriteJointPosition(Action):
    def __init__(self, env):
        super().__init__(env)
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
