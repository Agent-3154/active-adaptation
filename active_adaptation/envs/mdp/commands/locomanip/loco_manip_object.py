# HEADSUP: Never extract as method unless it is used in multiple places

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from typing_extensions import override

from active_adaptation.utils.math import (
    clamp_norm,
    quat_rotate,
    quat_rotate_inverse,
    wrap_to_pi,
    yaw_quat,
)
from active_adaptation.utils.symmetry import SymmetryTransform
from ..base import CommandV2

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.sensors import ContactSensor
    from active_adaptation.envs.env_base import EnvBase


class _LocoManipObjectBase(CommandV2):
    """Shared object spawn and layout for object-manipulation commands."""

    supported_backends = ("isaac",)

    def __init__(
        self,
        eef_body_name: str,
        object_name: str = "object",
        target_xy_range: Tuple[Tuple[float, float], Tuple[float, float]] | None = None,
    ) -> None:
        self.eef_body_name = eef_body_name
        self.object_name = object_name
        if target_xy_range is None:
            target_xy_range = ((-0.5, 0.5), (-0.5, 0.5))
        self.target_xy_range = target_xy_range

    @override
    def _initialize(self, env: "EnvBase") -> None:
        super()._initialize(env)

        body_ids, _ = self.asset.find_bodies(self.eef_body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected exactly one body matching {self.eef_body_name!r}, got {len(body_ids)}"
            )
        self.eef_body_idx = body_ids[0]

        self.object: RigidObject = self.env.scene[self.object_name]
        self.object_init_root_state = self.object.data.default_root_state.clone()

        self.contact_forces: ContactSensor = self.env.scene.sensors["contact_forces"]

        with torch.device(self.device):
            self.object_pos_w = torch.zeros(self.num_envs, 3)
            self.cmd_object_target_w = torch.zeros(self.num_envs, 3)
            self.cmd_object_target_b = torch.zeros(self.num_envs, 3)
            self.object_target_diff_w = torch.zeros(self.num_envs, 3)
            self.object_target_diff_b = torch.zeros(self.num_envs, 3)
            self.object_target_error_norm = torch.zeros(self.num_envs, 1)

    @property
    def eef_pos_w(self) -> torch.Tensor:
        return self.asset.data.body_link_pos_w[:, self.eef_body_idx]
    
    @property
    def eef_quat_w(self) -> torch.Tensor:
        return self.asset.data.body_link_quat_w[:, self.eef_body_idx]
    
    @property
    def eef_vel_w(self) -> torch.Tensor:
        return self.asset.data.body_link_lin_vel_w[:, self.eef_body_idx]

    @property
    def eef_pos_b(self) -> torch.Tensor:
        pos = quat_rotate_inverse(
            yaw_quat(self.asset.data.root_link_quat_w),
            self.eef_pos_w - self.asset.data.root_link_pos_w
        )
        pos[:, 2] = (
            self.eef_pos_w[:, 2]
            - self.env.get_ground_height_at(self.eef_pos_w)
        )
        return pos

    @staticmethod
    def _sample_uniform(
        num_samples: int, value_range: Tuple[float, float], device: torch.device
    ) -> torch.Tensor:
        lo, hi = value_range
        return torch.rand(num_samples, device=device) * (hi - lo) + lo

    @override
    def sample_init(self, env_ids: torch.Tensor) -> dict:
        origins = self.env.scene.get_spawn_origins(env_ids)
        n = len(env_ids)

        object_init = self.object_init_root_state[env_ids].clone()
        default_obj_z = object_init[:, 2].clone()
        object_init[:, 0] = origins[:, 0] + 2.0
        object_init[:, 1] = origins[:, 1] + 0.0
        object_init[:, 2] = (
            self.env.get_ground_height_at(object_init[:, :3]) + default_obj_z
        )
        if object_init.shape[-1] > 7:
            object_init[:, 7:] = 0.0

        robot_init = self.init_root_state[env_ids].clone()
        default_robot_z = robot_init[:, 2].clone()
        # y = self._sample_uniform(n, (-0.5, 0.5), self.device)
        robot_init[:, 0] = origins[:, 0]
        robot_init[:, 1] = origins[:, 1]
        robot_init[:, 2] = (
            self.env.get_ground_height_at(robot_init[:, :3]) + default_robot_z
        )
        return {"robot": robot_init, self.object_name: object_init}        


class LocoManipObject(_LocoManipObjectBase):
    """Training command: high-level object goal only (move object from A to B).

    The policy-facing command is the heading-frame object target and the
    object-to-target delta. No base velocity or EEF commands are issued; the
    policy must discover the required locomotion and manipulation.

    Layout (6D, body/yaw frame)::

        [target_x, target_y, target_z, diff_x, diff_y, diff_z]

    Horizontal components are yaw-aligned offsets from the root; ``target_z`` and
    ``diff_z`` use terrain-relative heights, matching the EEF command convention.
    """

    def command(self, key: str = "object") -> torch.Tensor:
        if key == "object":
            return torch.cat(
                [
                    self.object_pos_b,
                    self.cmd_object_target_b,
                    self.object_target_diff_b
                ],
                dim=-1,
            )
        raise ValueError(f"Invalid key: {key!r}; expected 'object'")

    @override
    def symmetry_transform(self, key: str = "object"):
        if key == "object":
            object_pos_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            cmd_object_target_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            object_target_diff_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            return SymmetryTransform.cat([
                object_pos_b,
                cmd_object_target_b,
                object_target_diff_b
            ])
        raise ValueError(f"Invalid key: {key!r}; expected 'object'")

    @override
    def reset(self, env_ids: torch.Tensor) -> None:
        obj_pos_w = self.object.data.root_link_pos_w[env_ids]
        offset = torch.zeros_like(obj_pos_w)
        offset[:, :2].uniform_(-1.0, 1.0)
        self.cmd_object_target_w[env_ids] = obj_pos_w + offset

    @override
    def sync_state(self) -> None:
        """Refresh object pose and body-frame target / error terms."""
        self.object_pos_w = self.object.data.root_pos_w
        self.root_pos_w = self.asset.data.root_link_pos_w
        self.root_yaw_q = yaw_quat(self.asset.data.root_link_quat_w)
        self.object_pos_b = quat_rotate_inverse(
            self.root_yaw_q,
            self.object_pos_w - self.root_pos_w
        )

        self.object_target_diff_w = self.cmd_object_target_w - self.object_pos_w
        self.object_target_diff_b = quat_rotate_inverse(
            self.root_yaw_q, self.object_target_diff_w)
        self.object_target_error_norm = self.object_target_diff_w.norm(dim=-1, keepdim=True)

    @override
    def update(self) -> None:
        pass

    @override
    def debug_draw(self) -> None:
        self.env.debug_draw.vector(
            self.object_pos_w,
            self.object_target_diff_w,
            color=(0.0, 0.0, 1.0, 1.0),
        )


class LocoManipObjectScripted(_LocoManipObjectBase):
    """Scripted playback command using the ``SingleEEFLocoManip`` interface.

    Open-loop phase schedule per env (by ``episode_length_buf``):
    approach → grasp → lift → move.
    """

    def __init__(
        self,
        eef_body_name: str,
        gripper_body_names: str,
        object_name: str = "object",
        grasp_height_range: Tuple[float, float] = (0.3, 0.5),
        target_xy_range: Tuple[Tuple[float, float], Tuple[float, float]] | None = None,
        standoff_distance: float = 0.65,
        standoff_linvel_gain: float = 2.0,
        standoff_yaw_gain: float = 1.0,
        speed_limit: float = 0.8,
        yaw_rate_range: Tuple[float, float] = (-1.0, 1.0),
        phase_approach_end: int = 200,
        phase_grasp_end: int = 400,
        phase_lift_end: int = 500,
    ) -> None:
        super().__init__(
            eef_body_name=eef_body_name,
            object_name=object_name,
            target_xy_range=target_xy_range,
        )

        self.gripper_body_names = gripper_body_names
        self.grasp_height_range = grasp_height_range
        self.standoff_distance = standoff_distance
        self.standoff_linvel_gain = standoff_linvel_gain
        self.standoff_yaw_gain = standoff_yaw_gain
        self.speed_limit = speed_limit
        self.yaw_rate_range = yaw_rate_range
        self.phase_approach_end = phase_approach_end
        self.phase_grasp_end = phase_grasp_end
        self.phase_lift_end = phase_lift_end

    @override
    def _initialize(self, env: "EnvBase") -> None:
        super()._initialize(env)

        body_ids, _ = self.asset.find_bodies(self.eef_body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected exactly one body matching {self.eef_body_name!r}, got {len(body_ids)}"
            )
        self.eef_body_idx = body_ids[0]

        gripper_body_ids, _ = self.contact_forces.find_bodies(self.gripper_body_names)
        if len(gripper_body_ids) != 2:
            raise ValueError(
                f"Expected exactly two bodies matching {self.gripper_body_names!r}, got {len(gripper_body_ids)}"
            )
        self.gripper_body_ids = torch.tensor(gripper_body_ids, device=self.device)

        with torch.device(self.device):
            self.grasp_height_per_env = torch.zeros(self.num_envs)
            self.grasp_point_w = torch.zeros(self.num_envs, 3)
            self.approach_standoff_w = torch.zeros(self.num_envs, 3)
            self.cmd_linvel_b = torch.zeros(self.num_envs, 3)
            self.cmd_linvel_w = torch.zeros(self.num_envs, 3)
            self.cmd_yawvel_b = torch.zeros(self.num_envs, 1)
            
            self.cmd_eef_pos_b = torch.zeros(self.num_envs, 3)
            self.cmd_eef_pos_w = torch.zeros(self.num_envs, 3)
            self.cmd_eef_rot_w = torch.zeros(self.num_envs, 4)

            self.cmd_eef_status = torch.zeros(self.num_envs, 1, dtype=torch.long)
            self.command_speed = torch.zeros(self.num_envs, 1)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            
            self.move_offset_w = torch.zeros(self.num_envs, 3)
            self.move_yaw = torch.zeros(self.num_envs, 1)
            
            # lift the object after grasping
            self._lift_offset = torch.zeros(self.num_envs, 3)
            self._lift_offset[:, 2].uniform_(0.05, 0.15)

            self.phase_ids = torch.zeros(self.num_envs, dtype=torch.long)
            self.should_grasp = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            self.grasp_time = torch.zeros(self.num_envs, 1, dtype=torch.float)
            # 0: approach and grasp
            # 1: grasp and lift
            # 2: lift and move

        if self.env.backend == "isaac" and self.env.sim.has_gui():
            from active_adaptation.envs.backends.isaac import IsaacSceneAdapter

            self.scene: IsaacSceneAdapter = self.env.scene
            self.grasp_point_marker = self.scene.create_sphere_marker(
                "/Visuals/Command/object_grasp_point",
                (1.0, 0.4, 0.0),
                radius=0.03
            )
            self.cmd_eef_pos_marker = self.scene.create_sphere_marker(
                "/Visuals/Command/cmd_eef_pos",
                (0.0, 0.4, 1.0),
                radius=0.05
            )
            self.eef_pose_marker = self.scene.create_frame_marker(
                "/Visuals/Command/target_eef_pose",
                scale=(0.1, 0.1, 0.1),
            )
        
        self.update()

    def command(self, key: str = "dense") -> torch.Tensor:
        if key == "dense":
            cmd = torch.cat([
                self.cmd_linvel_b[:, :2],
                self.cmd_yawvel_b,
                self.cmd_eef_pos_b,
                self.pos_diff_b,
                self.cmd_eef_forward_b,
                self.forward_diff_b,
                self.cmd_eef_upward_b,
                self.upward_diff_b,
                self.cmd_eef_status.float(),
                (1 - self.cmd_eef_status).float(),
            ], dim=-1)
            assert cmd.shape == (self.num_envs, 23)
            return cmd
        if key == "sparse":
            return torch.cat([
                self.cmd_eef_pos_b,
                self.pos_diff_b,
                self.cmd_eef_forward_b,
                self.forward_diff_b,
                self.cmd_eef_upward_b,
                self.upward_diff_b,
                self.cmd_eef_status.float(),
                (1 - self.cmd_eef_status).float(),
            ], dim=-1)
        raise ValueError(f"Invalid key: {key}")

    @override
    def symmetry_transform(self, key: str = "dense"):
        if key == "dense":
            return SymmetryTransform.cat([
                SymmetryTransform(perm=[0, 1], signs=[1, -1]),
                SymmetryTransform(perm=[0], signs=[-1]),
                SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1]),
                SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1]),
                SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1]),
                SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1]),
                SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1]),
                SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1]),
                SymmetryTransform(perm=[0, 1], signs=[1, 1]),
            ])
        if key == "sparse":
            return SymmetryTransform.cat([
                SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1]),
                SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1]),
                SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1]),
                SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1]),
                SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1]),
                SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1]),
                SymmetryTransform(perm=[0, 1], signs=[1, 1]),
            ])
        raise ValueError(f"Invalid key: {key}")

    @override
    def sample_init(self, env_ids: torch.Tensor) -> dict:
        init_state = super().sample_init(env_ids)
        self.grasp_height_per_env[env_ids] = self._sample_uniform(
            len(env_ids), self.grasp_height_range, self.device
        )
        return init_state        

    def sample_commands(self, env_ids: torch.Tensor) -> None:
        self.grasp_height_per_env[env_ids] = self._sample_uniform(
            len(env_ids), self.grasp_height_range, self.device
        )

    def _drive_base(
        self,
        env_ids: torch.Tensor,
        standoff_w: torch.Tensor,
        yaw_w: torch.Tensor,
    ) -> None:
        root_pos = self.asset.data.root_link_pos_w[env_ids]
        root_yaw_q = yaw_quat(self.asset.data.root_link_quat_w[env_ids])
        delta_w = standoff_w - root_pos
        delta_w[:, 2] = 0.0
        linvel_w = clamp_norm(
            self.standoff_linvel_gain * delta_w, max=self.speed_limit
        )
        self.cmd_linvel_w[env_ids] = linvel_w
        self.cmd_linvel_b[env_ids] = quat_rotate_inverse(root_yaw_q, linvel_w)
        yaw_err = wrap_to_pi(yaw_w.reshape(-1) - self.asset.data.heading_w[env_ids])
        self.cmd_yawvel_b[env_ids, 0] = (
            self.standoff_yaw_gain * yaw_err
        ).clamp(*self.yaw_rate_range)

    def _phase_approach(self, env_ids: torch.Tensor) -> None:
        root_pos_w = self.asset.data.root_link_pos_w[env_ids]
        root_yaw_q = self.root_yaw_quat[env_ids]
        grasp_point = self.grasp_point_w[env_ids]

        self.cmd_eef_pos_w[env_ids] = root_pos_w + clamp_norm(grasp_point - root_pos_w, max=0.6)
        self.cmd_eef_pos_b[env_ids] = quat_rotate_inverse(
            root_yaw_q,
            self.cmd_eef_pos_w[env_ids] - root_pos_w * torch.tensor([1.0, 1.0, 0.0], device=self.device)
        )

        self.cmd_eef_rot_w[env_ids] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)
        self._drive_base(
            env_ids,
            self.approach_standoff_w[env_ids],
            torch.zeros(len(env_ids), device=self.device),
        )

        eef_pos_error = (grasp_point[:, :2] - self.eef_pos_w[env_ids, :2]).norm(dim=-1, keepdim=True)
        should_grasp = (eef_pos_error < 0.02).reshape(-1, 1)
        self.should_grasp[env_ids] = should_grasp | self.should_grasp[env_ids]
        self.cmd_eef_status[env_ids] = torch.where(should_grasp, 1, 0)
        ct = self.contact_forces.data.current_contact_time[env_ids][:, self.gripper_body_ids]
        ct = ct.amax(dim=-1, keepdim=True)
        
        next_phase = torch.where(should_grasp & (ct > 0.5), 1, 0)
        self.phase_ids[env_ids] = next_phase.squeeze(-1)

    def _phase_lift(self, env_ids: torch.Tensor) -> None:
        root_pos_w = self.asset.data.root_link_pos_w[env_ids]
        root_yaw_q = self.root_yaw_quat[env_ids]
        lift_target = self.grasp_point_w[env_ids] + self._lift_offset[env_ids]
    
        self.cmd_eef_pos_w[env_ids] = lift_target
        self.cmd_eef_pos_b[env_ids] = quat_rotate_inverse(
            root_yaw_q,
            lift_target - root_pos_w * torch.tensor([1.0, 1.0, 0.0], device=self.device)
        )
        self.cmd_eef_rot_w[env_ids] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)
        self._drive_base(
            env_ids,
            self.approach_standoff_w[env_ids],
            torch.zeros(len(env_ids), device=self.device),
        )        
        self.cmd_eef_status[env_ids, 0] = 1
        ct = self.contact_forces.data.current_contact_time[env_ids][:, self.gripper_body_ids]
        ct = ct.amax(dim=-1, keepdim=True)

        next_phase = torch.where(ct > 1.0, 2, 1)
        self.phase_ids[env_ids] = next_phase.squeeze(-1)

    def _phase_move(self, env_ids: torch.Tensor) -> None:
        # maintain EEF hold pose in the body frame
        # and compute world frame from body frame
        self.cmd_eef_pos_w[env_ids] = (
            self.root_pos_w[env_ids] * torch.tensor([1.0, 1.0, 0.0], device=self.device) +
            + quat_rotate(self.root_yaw_quat[env_ids], self.cmd_eef_pos_b[env_ids])
        )

        self._drive_base(
            env_ids,
            self.approach_standoff_w[env_ids] + self.move_offset_w[env_ids],
            self.move_yaw[env_ids],
        )
        self.cmd_eef_status[env_ids, 0] = 1

    @override
    def reset(self, env_ids: torch.Tensor) -> None:
        self.sample_commands(env_ids)
        # compute standoff position # do not extract as method
        robot_w = self.asset.data.root_link_pos_w[env_ids]
        object_w = self.object.data.root_pos_w[env_ids]
        diff = robot_w - object_w
        direction = diff / diff.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        standoff = object_w + direction * self.standoff_distance
        standoff[:, 2] = self.env.get_ground_height_at(standoff)
        self.approach_standoff_w[env_ids] = standoff

        move_offset = torch.zeros(len(env_ids), 3, device=self.device)
        move_offset[:, 0].uniform_(-1.0, 1.0)
        move_offset[:, 1].uniform_(-2.0, 2.0)
        move_yaw = torch.zeros(len(env_ids), 1, device=self.device)
        move_yaw.uniform_(-torch.pi / 2, torch.pi / 2)
        self.move_offset_w[env_ids] = move_offset
        self.move_yaw[env_ids] = move_yaw

        self.phase_ids[env_ids] = 0 # reset to approach phase
        self.should_grasp[env_ids] = False

    def _read_robot_and_object_state(self) -> None:
        self.root_pos_w = self.asset.data.root_link_pos_w
        self.root_yaw_quat = yaw_quat(self.asset.data.root_link_quat_w)
        self.object_pos_w = self.object.data.root_pos_w
        self.object_quat_w = self.object.data.root_quat_w
        offset_obj = torch.zeros(self.num_envs, 3, device=self.device)
        offset_obj[:, 2] = self.grasp_height_per_env
        self.grasp_point_w = self.object_pos_w + quat_rotate(self.object_quat_w, offset_obj)

    def _sync_command_orientation(self) -> None:
        self.cmd_eef_forward_w = quat_rotate(
            self.cmd_eef_rot_w,
            torch.tensor([[1.0, 0.0, 0.0]], device=self.device),
        )
        self.cmd_eef_upward_w = quat_rotate(
            self.cmd_eef_rot_w,
            torch.tensor([[0.0, 0.0, 1.0]], device=self.device),
        )
        self.cmd_eef_forward_b = quat_rotate_inverse(
            self.root_yaw_quat,
            self.cmd_eef_forward_w,
        )
        self.cmd_eef_upward_b = quat_rotate_inverse(
            self.root_yaw_quat,
            self.cmd_eef_upward_w,
        )

    def _compute_tracking_errors(self) -> None:
        forward_axis_b = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)
        upward_axis_b = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)
        self.eef_forward_w = quat_rotate(self.eef_quat_w, forward_axis_b)
        self.eef_forward_b = quat_rotate_inverse(self.root_yaw_quat, self.eef_forward_w)
        self.eef_upward_w = quat_rotate(self.eef_quat_w, upward_axis_b)
        self.eef_upward_b = quat_rotate_inverse(self.root_yaw_quat, self.eef_upward_w)

        self.pos_diff_w = self.cmd_eef_pos_w - self.eef_pos_w
        self.pos_diff_b = quat_rotate_inverse(
            yaw_quat(self.asset.data.root_link_quat_w),
            self.pos_diff_w,
        )
        self.forward_diff_w = self.cmd_eef_forward_w - self.eef_forward_w
        self.forward_diff_b = quat_rotate_inverse(
            self.root_yaw_quat,
            self.forward_diff_w,
        )
        self.upward_diff_w = self.cmd_eef_upward_w - self.eef_upward_w
        self.upward_diff_b = quat_rotate_inverse(
            self.root_yaw_quat,
            self.upward_diff_w,
        )

        self.command_speed = self.cmd_linvel_w.norm(dim=-1, keepdim=True)
        self.is_standing_env = self.command_speed < 0.1

    @override
    def sync_state(self) -> None:
        self._read_robot_and_object_state()
        self._sync_command_orientation()
        self._compute_tracking_errors()

    @override
    def update(self) -> None:
        self._read_robot_and_object_state()

        approach_ids = (self.phase_ids == 0).nonzero(as_tuple=False).squeeze(-1)
        self._phase_approach(approach_ids)
        lift_ids = (self.phase_ids == 1).nonzero(as_tuple=False).squeeze(-1)
        self._phase_lift(lift_ids)
        move_ids = (self.phase_ids == 2).nonzero(as_tuple=False).squeeze(-1)
        self._phase_move(move_ids)

        self._sync_command_orientation()
        self._compute_tracking_errors()

    @override
    def debug_draw(self) -> None:
        self.env.debug_draw.vector(
            self.asset.data.root_link_pos_w,
            self.cmd_linvel_w,
            color=(1.0, 1.0, 1.0, 1.0),
        )
        self.env.debug_draw.vector(
            self.eef_pos_w, self.eef_forward_w, color=(1.0, 0.0, 0.0, 1.0)
        )
        self.env.debug_draw.vector(
            self.eef_pos_w,
            self.cmd_eef_pos_w - self.eef_pos_w,
            color=(0.0, 0.0, 1.0, 1.0),
        )
        self.grasp_point_marker.visualize(self.grasp_point_w)
        self.cmd_eef_pos_marker.visualize(self.cmd_eef_pos_w)

        self.eef_pose_marker.visualize(
            translations=self.cmd_eef_pos_w,
            orientations=self.cmd_eef_rot_w,
        )


__all__ = ["LocoManipObject", "LocoManipObjectScripted"]
