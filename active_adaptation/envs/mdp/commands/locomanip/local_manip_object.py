from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from typing_extensions import override

from active_adaptation.utils.math import (
    euler_from_quat,
    euler_rotate,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
    sample_quat_yaw,
    wrap_to_pi,
    yaw_quat,
)
from active_adaptation.utils.symmetry import SymmetryTransform
from ..base import Command

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject


class LocalManipObject(Command):
    """Object-aware loco-manip command with the `SingleEEFLocoManip` command layout."""

    supported_backends = ("isaac",)

    def __init__(
        self,
        env,
        eef_body_name: str,
        object_name: str = "object",
        object_distance_range: Tuple[float, float] = (2.0, 3.0),
        grasp_height_range: Tuple[float, float] = (0.05, 0.6),
        standoff_distance_range: Tuple[float, float] = (0.6, 0.8),
        standoff_angle_range: Tuple[float, float] = (-torch.pi / 3, torch.pi / 3),
        linvel_x_range: Tuple[float, float] = (-1.0, 1.0),
        linvel_y_range: Tuple[float, float] = (-1.0, 1.0),
        yaw_rate_range: Tuple[float, float] = (-1.0, 1.0),
        standoff_linvel_gain: float = 1.0,
        standoff_yaw_gain: float = 1.0,
        teleop: bool = False,
    ) -> None:
        super().__init__(env, teleop)
        body_ids, _ = self.asset.find_bodies(eef_body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected exactly one body matching {eef_body_name!r}, got {body_ids.numel()}"
            )
        self.eef_body_idx = body_ids[0]

        self.object_name = object_name
        self.object: RigidObject = self.env.scene[object_name]
        self.object_init_root_state = self.object.data.default_root_state.clone()
        self.object_distance_range = object_distance_range
        self.grasp_height_range = grasp_height_range
        self.standoff_distance_range = standoff_distance_range
        self.standoff_angle_range = standoff_angle_range
        self.linvel_x_range = linvel_x_range
        self.linvel_y_range = linvel_y_range
        self.yaw_rate_range = yaw_rate_range
        self.standoff_linvel_gain = standoff_linvel_gain
        self.standoff_yaw_gain = standoff_yaw_gain

        with torch.device(self.device):
            self.cmd_linvel_b = torch.zeros(self.num_envs, 3)
            self.cmd_linvel_w = torch.zeros(self.num_envs, 3)
            self.cmd_yawvel_b = torch.zeros(self.num_envs, 1)
            self.cmd_eef_pos_b = torch.zeros(self.num_envs, 3)
            self.cmd_eef_pos_w = torch.zeros(self.num_envs, 3)
            self.cmd_eef_pitch_w = torch.zeros(self.num_envs, 1)
            self.eef_forward_w = torch.zeros(self.num_envs, 3)
            self.cmd_eef_forward_w = torch.zeros(self.num_envs, 3)
            self.cmd_eef_vel_b = torch.zeros(self.num_envs, 3)
            self.cmd_eef_vel_w = torch.zeros(self.num_envs, 3)
            self.grasp_height = torch.zeros(self.num_envs)
            self.grasp_point_w = torch.zeros(self.num_envs, 3)
            self.standoff_pos_w = torch.zeros(self.num_envs, 3)
            self.standoff_yaw_w = torch.zeros(self.num_envs, 1)
            self.command_speed = torch.zeros(self.num_envs, 1)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=torch.bool)

        self.grasp_marker = None
        self.standoff_marker = None
        if self.env.backend == "isaac" and self.env.sim.has_gui():
            from active_adaptation.envs.backends.isaac import IsaacSceneAdapter

            self.scene: IsaacSceneAdapter = self.env.scene
            self.grasp_marker = self.scene.create_sphere_marker(
                "/Visuals/Command/object_grasp_point",
                (1.0, 0.4, 0.0),
                radius=0.03,
            )
            self.standoff_marker = self.scene.create_sphere_marker(
                "/Visuals/Command/object_standoff_pos",
                (0.0, 0.7, 1.0),
                radius=0.04,
            )
    
    @property
    def command(self) -> torch.Tensor:
        pos_diff_w = self.cmd_eef_pos_w - self.eef_pos_w
        pos_diff_b = quat_rotate_inverse(
            yaw_quat(self.asset.data.root_link_quat_w),
            pos_diff_w
        )
        return torch.cat(
            [
                self.cmd_linvel_b[:, :2],
                self.cmd_yawvel_b,
                self.cmd_eef_pos_b,
                pos_diff_b,
                self.cmd_eef_pitch_w,
            ],
            dim=-1,
        )
    
    @override
    def symmetry_transform(self):
        cmd_linvel_b = SymmetryTransform(perm=[0, 1], signs=[1, -1])
        cmd_yawvel_b = SymmetryTransform(perm=[0], signs=[-1])
        cmd_eef_pos_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
        cmd_eef_pitch_w = SymmetryTransform(perm=[0], signs=[1])
        return SymmetryTransform.cat(
            [cmd_linvel_b, cmd_yawvel_b, cmd_eef_pos_b, cmd_eef_pitch_w]
        )

    @override
    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor | None:
        origins = self.env.scene.get_spawn_origins(env_ids)

        robot_init = self.init_root_state[env_ids].clone()
        robot_init[:, :3] += origins
        robot_init[:, 3:7] = quat_mul(
            robot_init[:, 3:7],
            sample_quat_yaw(len(env_ids), device=self.device),
        )

        object_init = self.object_init_root_state[env_ids].clone()
        object_angle = torch.rand(len(env_ids), device=self.device) * 2 * torch.pi
        object_dist = self._sample_uniform(len(env_ids), self.object_distance_range)
        object_init[:, 0] = origins[:, 0] + object_dist * torch.cos(object_angle)
        object_init[:, 1] = origins[:, 1] + object_dist * torch.sin(object_angle)
        object_init[:, 2] += self.env.get_ground_height_at(object_init[:, :3])
        object_init[:, 3:7] = sample_quat_yaw(len(env_ids), device=self.device)
        if object_init.shape[-1] > 7:
            object_init[:, 7:] = 0.0

        return {
            "robot": robot_init,
            self.object_name: object_init,
        }

    def _sample_uniform(
        self, num_samples: int, value_range: Tuple[float, float]
    ) -> torch.Tensor:
        return (
            torch.rand(num_samples, device=self.device)
            * (value_range[1] - value_range[0])
            + value_range[0]
        )
    
    def sample_commands(self, env_ids: torch.Tensor) -> None:
        self.grasp_height[env_ids] = self._sample_uniform(
            len(env_ids), self.grasp_height_range
        )

        obj_pos_w = self.object.data.root_pos_w[env_ids]
        origins = self.env.scene.env_origins[env_ids]
        to_origin = origins[:, :2] - obj_pos_w[:, :2]
        to_origin = to_origin / to_origin.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        base_angle = torch.atan2(to_origin[:, 1], to_origin[:, 0])
        angle = base_angle + self._sample_uniform(len(env_ids), self.standoff_angle_range)
        distance = self._sample_uniform(len(env_ids), self.standoff_distance_range)
        standoff_pos_w = obj_pos_w.clone()
        standoff_pos_w[:, 0] = obj_pos_w[:, 0] + distance * torch.cos(angle)
        standoff_pos_w[:, 1] = obj_pos_w[:, 1] + distance * torch.sin(angle)
        standoff_pos_w[:, 2] = self.env.get_ground_height_at(standoff_pos_w)

        self.standoff_pos_w[env_ids] = standoff_pos_w
        self.standoff_yaw_w[env_ids, 0] = torch.atan2(
            obj_pos_w[:, 1] - standoff_pos_w[:, 1],
            obj_pos_w[:, 0] - standoff_pos_w[:, 0],
        )
        self.sample_eef_pitch_commands(env_ids, self.grasp_height[env_ids])

    def sample_eef_pitch_commands(
        self, env_ids: torch.Tensor, target_height: torch.Tensor
    ) -> None:
        pitch_down = torch.rand(len(env_ids), device=self.device) * (torch.pi / 2)
        pitch_up = torch.rand(len(env_ids), device=self.device) * (torch.pi / 2) - (
            torch.pi / 2
        )
        self.cmd_eef_pitch_w[env_ids, 0] = torch.where(
            target_height < 0.4,
            pitch_down,
            pitch_up,
        )

    @override
    def reset(self, env_ids: torch.Tensor) -> None:
        self.sample_commands(env_ids)
        self.update()
    
    @override
    def update(self) -> None:
        obj_pos_w = self.object.data.root_pos_w
        obj_quat_w = self.object.data.root_quat_w
        grasp_offset_obj = torch.zeros(self.num_envs, 3, device=self.device)
        grasp_offset_obj[:, 2] = self.grasp_height
        self.grasp_point_w = quat_rotate(obj_quat_w, grasp_offset_obj) + obj_pos_w

        root_pos = self.asset.data.root_link_pos_w
        root_yaw_q = yaw_quat(self.asset.data.root_link_quat_w)

        eef_delta_w = self.grasp_point_w - root_pos
        eef_delta_w[:, 2] = 0.0
        eef_delta_b = quat_rotate_inverse(root_yaw_q, eef_delta_w)
        self.cmd_eef_pos_b[:, :2] = eef_delta_b[:, :2]
        self.cmd_eef_pos_b[:, 2] = (
            self.grasp_point_w[:, 2] - self.env.get_ground_height_at(self.grasp_point_w)
        )
        self.cmd_eef_pos_w = self.grasp_point_w.clone()
        self.eef_pos_w = self.asset.data.body_link_pos_w[:, self.eef_body_idx]

        standoff_delta_w = self.standoff_pos_w - root_pos
        standoff_delta_w[:, 2] = 0.0
        standoff_delta_b = quat_rotate_inverse(root_yaw_q, standoff_delta_w)
        self.cmd_linvel_b[:, 0] = (
            self.standoff_linvel_gain * standoff_delta_b[:, 0]
        ).clamp(*self.linvel_x_range)
        self.cmd_linvel_b[:, 1] = (
            self.standoff_linvel_gain * standoff_delta_b[:, 1]
        ).clamp(*self.linvel_y_range)
        self.cmd_linvel_b[:, 2] = 0.0
        self.cmd_linvel_w = quat_rotate(root_yaw_q, self.cmd_linvel_b)

        yaw_error = wrap_to_pi(
            self.standoff_yaw_w - self.asset.data.heading_w.unsqueeze(1)
        )
        self.cmd_yawvel_b = (self.standoff_yaw_gain * yaw_error).clamp(
            *self.yaw_rate_range
        )

        forward_axis_b = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)
        eef_quat_w = self.asset.data.body_link_quat_w[:, self.eef_body_idx]
        eef_rpy_w = euler_from_quat(eef_quat_w)
        eef_rpy_w[:, 1] = self.cmd_eef_pitch_w.squeeze(-1)
        self.eef_forward_w = quat_rotate(eef_quat_w, forward_axis_b)
        self.cmd_eef_forward_w = euler_rotate(eef_rpy_w, forward_axis_b)

        self.command_speed = self.cmd_linvel_w.norm(dim=-1, keepdim=True)
        self.is_standing_env = self.command_speed < 0.1
    
    @override
    def debug_draw(self) -> None:
        self.env.debug_draw.vector(
            self.asset.data.root_link_pos_w,
            self.cmd_linvel_w,
            color=(1.0, 1.0, 1.0, 1.0),
        )
        self.env.debug_draw.vector(
            self.asset.data.body_link_pos_w[:, self.eef_body_idx],
            self.eef_forward_w,
            color=(1.0, 0.0, 0.0, 1.0),
        )
        self.env.debug_draw.vector(
            self.asset.data.body_link_pos_w[:, self.eef_body_idx],
            self.cmd_eef_forward_w,
            color=(0.0, 1.0, 0.0, 1.0),
        )
        self.env.debug_draw.vector(
            self.eef_pos_w,
            self.cmd_eef_pos_w - self.eef_pos_w,
            color=(0.0, 0.0, 1.0, 1.0),
        )
        if self.grasp_marker is not None:
            self.grasp_marker.visualize(self.grasp_point_w)
        if self.standoff_marker is not None:
            self.standoff_marker.visualize(self.standoff_pos_w)


__all__ = ["LocalManipObject"]