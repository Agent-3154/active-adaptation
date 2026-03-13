from __future__ import annotations

import torch

from typing import Dict, Literal, Optional, Tuple
from typing_extensions import override

try:
    import isaaclab.utils.string as string_utils
except ModuleNotFoundError:
    from mjlab.utils.lab_api import string as string_utils

from active_adaptation.assets import get_input_joint_indexing
from active_adaptation.utils.symmetry import joint_space_symmetry

from .base import Action


class _DelayedJointAction(Action):
    def __init__(
        self,
        env,
        action_scaling: Dict[str, float] = 0.5,
        max_delay: int = 2,
        alpha_range: Tuple[float, float] = (0.5, 1.0),
        input_order: Literal["isaac", "mujoco", "mjlab"] = "isaac",
    ):
        super().__init__(env)
        action_scaling = dict(action_scaling)
        self.joint_ids, self.joint_names, self.action_scaling = self.resolve(action_scaling)
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)

        self.indexing, self.input_joint_names = get_input_joint_indexing(
            input_order=input_order,
            asset_cfg=self.asset.cfg,
            target_joint_names=self.joint_names,
            device=self.device,
        )
        self.action_scaling = torch.tensor(self.action_scaling, device=self.device)
        self.max_delay = max_delay
        self.alpha_range = tuple(alpha_range)
        self.decimation = int(self.env.step_dt / self.env.physics_dt)

        with torch.device(self.device):
            self.action_buf = torch.zeros(
                self.num_envs, 4, self.action_dim, device=self.device
            )
            self.action_queue = torch.zeros(
                self.num_envs,
                self.max_delay + self.decimation,
                self.action_dim,
                device=self.device,
            )
            self.applied_action = torch.zeros(
                self.num_envs, self.action_dim, device=self.device
            )
            self.alpha = torch.ones(self.num_envs, 1, device=self.device)
            self.delay = torch.zeros(
                self.num_envs, 1, dtype=torch.int64, device=self.device
            )

    @property
    def action_dim(self):
        return len(self.joint_ids)

    def resolve(self, spec):
        return string_utils.resolve_matching_names_values(
            dict(spec), self.asset.joint_names
        )

    @override
    def reset(self, env_ids: torch.Tensor):
        self.delay[env_ids] = torch.randint(
            0, self.max_delay + 1, (len(env_ids), 1), device=self.device
        )
        self.action_buf[env_ids] = 0
        self.applied_action[env_ids] = 0

        alpha = torch.empty(len(env_ids), 1, device=self.device)
        alpha.uniform_(self.alpha_range[0], self.alpha_range[1])
        self.alpha[env_ids] = alpha

    @override
    def process_action(self, action: Optional[torch.Tensor]):
        if action is None:
            return

        action = action[:, self.indexing]
        self.action_buf = self.action_buf.roll(1, dims=1)
        self.action_buf[:, 0] = action
        delay_mask = (
            torch.arange(self.action_queue.shape[1], device=self.device)
            < self.delay
        ).reshape(self.num_envs, self.action_queue.shape[1], 1)
        self.action_queue = torch.where(delay_mask, self.action_queue, action.unsqueeze(1))

    @override
    def symmetry_transform(self):
        return joint_space_symmetry(self.asset, self.input_joint_names)


class JointPosition(_DelayedJointAction):
    def __init__(
        self,
        env,
        action_scaling: Dict[str, float] = 0.5,
        max_delay: int = 2,
        alpha_range: Tuple[float, float] = (0.5, 1.0),
        input_order: Literal["isaac", "mujoco", "mjlab"] = "isaac",
    ):
        super().__init__(
            env,
            action_scaling=action_scaling,
            max_delay=max_delay,
            alpha_range=alpha_range,
            input_order=input_order,
        )
        self.default_joint_pos = self.asset.data.default_joint_pos.clone()
        self.offset = torch.zeros_like(self.default_joint_pos)

    @override
    def reset(self, env_ids: torch.Tensor):
        super().reset(env_ids)
        default_joint_pos = self.asset.data.default_joint_pos[env_ids]
        self.default_joint_pos[env_ids] = default_joint_pos + self.offset[env_ids]

    @override
    def apply_action(self, substep: int):
        self.applied_action.lerp_(self.action_queue[:, 0], self.alpha)
        self.action_queue = self.action_queue.roll(-1, dims=1)

        jpos_target = self.default_joint_pos.clone()
        jpos_target[:, self.joint_ids] += self.applied_action * self.action_scaling
        self.asset.set_joint_position_target(jpos_target)


class JointVelocity(_DelayedJointAction):
    @override
    def apply_action(self, substep: int):
        self.applied_action.lerp_(self.action_queue[:, 0], self.alpha)
        self.action_queue = self.action_queue.roll(-1, dims=1)

        jvel_target = self.applied_action * self.action_scaling
        self.asset.set_joint_velocity_target(jvel_target, joint_ids=self.joint_ids)


__all__ = ["JointPosition", "JointVelocity"]
