# HEADSUP: Never extract as method unless it is used in multiple places

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from typing_extensions import override
from tensordict import TensorDict

from active_adaptation.utils.math import (
    clamp_norm,
    quat_rotate,
    quat_rotate_inverse,
    wrap_to_pi,
    yaw_quat,
)
from active_adaptation.utils.symmetry import SymmetryTransform
from active_adaptation.envs.mdp.commands.base import CommandV2
from active_adaptation.envs.mdp.rewards.base import RewardV2

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
        gripper_joint_names: str,
        gripper_body_names: str,
        object_name: str = "object",
    ) -> None:
        self.eef_body_name = eef_body_name
        self.object_name = object_name
        self.gripper_joint_names = gripper_joint_names
        self.gripper_body_names = gripper_body_names

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

        self.gripper_joint_ids, _ = self.asset.find_joints(self.gripper_joint_names)
        self.gripper_joint_ids = torch.tensor(self.gripper_joint_ids, device=self.device)
        limits = self.asset.data.soft_joint_pos_limits[0, self.gripper_joint_ids]
        self._gripper_max_open = limits.abs().amax(dim=-1).max().clamp_min(1e-6)

        gripper_body_ids, _ = self.contact_forces.find_bodies(self.gripper_body_names)
        if len(gripper_body_ids) != 2:
            raise ValueError(
                f"Expected exactly two bodies matching {self.gripper_body_names!r}, got {len(gripper_body_ids)}"
            )
        self.gripper_body_ids = torch.tensor(gripper_body_ids, device=self.device)

        with torch.device(self.device):
            self.object_pos_w = torch.zeros(self.num_envs, 3)
            self.cmd_object_target_w = torch.zeros(self.num_envs, 3)
            self.cmd_object_target_b = torch.zeros(self.num_envs, 3)
            self.cmd_object_vel_w = torch.zeros(self.num_envs, 3)
            
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=torch.bool)

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
        xy_offset = torch.zeros(n, 2, device=self.device)
        xy_offset[:, 0].uniform_(1.2, 2.0)
        object_init[:, :2] = xy_offset + origins[:, :2]
        object_init[:, 7:] = 0.0

        robot_init = self.init_root_state[env_ids].clone()
        default_robot_z = robot_init[:, 2].clone()
        # y = self._sample_uniform(n, (-0.5, 0.5), self.device)
        robot_init[:, :2] = origins[:, :2]
        robot_init[:, 2] = (
            self.env.get_ground_height_at(robot_init[:, :3]) + default_robot_z
        )
        return {"robot": robot_init, self.object_name: object_init}     

    def get_gripper_status(self) -> torch.Tensor:
        """Return gripper closedness in ``[0, 1]`` (0=open, 1=closed)."""
        gripper_pos = self.asset.data.joint_pos[:, self.gripper_joint_ids]
        openness = (
            gripper_pos.abs().amax(dim=-1, keepdim=True) / self._gripper_max_open
        ).clamp(0.0, 1.0)
        joint_closedness = 1.0 - openness

        ct = self.contact_forces.data.current_contact_time[:, self.gripper_body_ids]
        both_in_contact = ct.gt(0.0).all(dim=-1, keepdim=True).float()
        return joint_closedness.maximum(both_in_contact)


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

    def __init__(
        self,
        eef_body_name: str,
        gripper_joint_names: str,
        gripper_body_names: str,
        object_name: str = "object",
        target_offset_range: Tuple[float, float] = (-2.0, 2.0),
        object_vel_gain: float = 1.0,
        object_vel_limit: float = 0.5,
        grasp_height_range: Tuple[float, float] = (0.3, 0.6),
    ) -> None:
        super().__init__(
            eef_body_name=eef_body_name,
            object_name=object_name,
            gripper_joint_names=gripper_joint_names,
            gripper_body_names=gripper_body_names,
        )
        self.target_offset_range = target_offset_range
        self.object_vel_gain = object_vel_gain
        self.object_vel_limit = object_vel_limit
        self.grasp_height_range = grasp_height_range
    
    @override
    def _initialize(self, env: "EnvBase") -> None:
        super()._initialize(env)
        self.grasp_height_per_env = torch.zeros(self.num_envs, device=self.device)
        self.sync_state()

        if self.env.backend == "isaac" and self.env.sim.has_gui():
            from active_adaptation.envs.backends.isaac import IsaacSceneAdapter
            self.scene: IsaacSceneAdapter = self.env.scene
            self.grasp_point_marker = self.scene.create_sphere_marker(
                "/Visuals/Command/object_grasp_point",
                (1.0, 0.4, 0.0),
                radius=0.03
            )

    def command(self, key: str = "object") -> torch.Tensor:
        if key == "object":
            return torch.cat(
                [
                    self.object_pos_b,
                    self.object_vel_b,
                    self.grasp_point_b,
                    self.grasp_point_diff_b,
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
            object_vel_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            grasp_point_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            grasp_point_diff_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            cmd_object_target_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            object_target_diff_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            return SymmetryTransform.cat([
                object_pos_b,
                object_vel_b,
                grasp_point_b,
                grasp_point_diff_b,
                cmd_object_target_b,
                object_target_diff_b
            ])
        raise ValueError(f"Invalid key: {key!r}; expected 'object'")

    @override
    def reset(self, env_ids: torch.Tensor) -> None:
        self._sample_target(env_ids)
        self.grasp_height_per_env[env_ids] = self._sample_uniform(len(env_ids), self.grasp_height_range, self.device)
    
    def _sample_target(self, env_ids: torch.Tensor) -> None:
        obj_pos_w = self.object.data.root_link_pos_w[env_ids]
        offset = torch.zeros_like(obj_pos_w)
        offset[:, :2].uniform_(self.target_offset_range[0], self.target_offset_range[1])
        self.cmd_object_target_w[env_ids] = obj_pos_w + offset

    @override
    def sync_state(self) -> None:
        """Refresh object pose and body-frame target / error terms."""
        self.object_pos_w = self.object.data.root_pos_w
        self.object_quat_w = self.object.data.root_quat_w
        self.object_vel_w = self.object.data.root_lin_vel_w

        self.root_pos_w = self.asset.data.root_link_pos_w
        self.root_yaw_q = yaw_quat(self.asset.data.root_link_quat_w)
        
        self.object_pos_b = quat_rotate_inverse(
            self.root_yaw_q,
            self.object_pos_w - self.root_pos_w
        )
        self.object_vel_b = quat_rotate_inverse(
            self.root_yaw_q,
            self.object_vel_w,
        )
        self.cmd_object_target_b = quat_rotate_inverse(
            self.root_yaw_q,
            self.cmd_object_target_w - self.root_pos_w,
        )

        self.object_target_diff_w = self.cmd_object_target_w - self.object_pos_w
        self.object_target_diff_b = quat_rotate_inverse(
            self.root_yaw_q, self.object_target_diff_w)
        self.object_target_error_norm = self.object_target_diff_w.norm(dim=-1, keepdim=True)
        offset_obj = torch.zeros(self.num_envs, 3, device=self.device)
        offset_obj[:, 2] = self.grasp_height_per_env
        self.grasp_point_w = self.object_pos_w + quat_rotate(self.object_quat_w, offset_obj)
        self.grasp_point_b = quat_rotate_inverse(
            self.root_yaw_q,
            self.grasp_point_w - self.root_pos_w
        )
        self.grasp_point_diff_w = self.grasp_point_w - self.eef_pos_w
        self.grasp_point_diff_b = quat_rotate_inverse(
            self.root_yaw_q,
            self.grasp_point_diff_w
        )

    @override
    def update(self) -> None:
        self.cmd_object_vel_w = clamp_norm(
            self.object_vel_gain * self.object_target_diff_w,
            max=self.object_vel_limit,
        )

    @override
    def debug_draw(self) -> None:
        self.env.debug_draw.vector(
            self.object_pos_w,
            self.object_target_diff_w,
            color=(0.0, 0.0, 1.0, 1.0),
        )
        self.env.debug_draw.vector(
            self.object_pos_w,
            self.cmd_object_vel_w,
            color=(1.0, 0.0, 0.0, 1.0),
        )
        self.grasp_point_marker.visualize(self.grasp_point_w)
        self.env.debug_draw.vector(
            self.eef_pos_w,
            self.grasp_point_diff_w,
            color=(0.0, 1.0, 0.0, 1.0),
        )
    
    @override
    def relabel_command(self, tensordict: TensorDict) -> TensorDict:
        """
        Use the final object pose in a trajectory as the target pose.
        """
        T, N = tensordict.shape[:2]
        root_state_w = tensordict["command_state", "root_state_w"]
        root_yaw_q = yaw_quat(root_state_w[..., 3:7])
        eef_state_w = tensordict["command_state", "eef_state_w"]
        eef_pos_w = eef_state_w[..., :3]
        object_state_w = tensordict["command_state", "object_state_w"]
        object_pos_w = object_state_w[..., :3]
        object_vel_w = object_state_w[..., 7:10]
        # object_quat_w = object_state_w[..., 3:7] # unused for now
        object_pos_b = quat_rotate_inverse(
            root_yaw_q,
            object_pos_w - root_state_w[..., :3]
        )
        object_vel_b = quat_rotate_inverse(
            root_yaw_q,
            object_vel_w,
        )
        object_final_pos_w = object_pos_w[-1]
        cmd_object_target_w = object_final_pos_w.expand(T, N, 3)
        cmd_object_target_b = quat_rotate_inverse(
            root_yaw_q,
            cmd_object_target_w - root_state_w[..., :3]
        )
        object_target_diff_w = cmd_object_target_w - object_pos_w
        object_target_diff_b = quat_rotate_inverse(
            root_yaw_q,
            object_target_diff_w
        )
        object_target_error_norm = object_target_diff_w.norm(dim=-1, keepdim=True)
        cmd_object_vel_w = clamp_norm(
            self.object_vel_gain * object_target_diff_w,
            max=self.object_vel_limit,
        )
        grasp_point_w = tensordict["command_state", "grasp_point_w"]
        grasp_point_b = quat_rotate_inverse(
            root_yaw_q,
            grasp_point_w - root_state_w[..., :3]
        )
        grasp_point_diff_w = grasp_point_w - eef_pos_w
        grasp_point_diff_b = quat_rotate_inverse(
            root_yaw_q,
            grasp_point_diff_w
        )
        command = torch.cat([
            object_pos_b,
            object_vel_b,
            grasp_point_b,
            grasp_point_diff_b,
            cmd_object_target_b,
            object_target_diff_b
        ], dim=-1)
        tensordict["command"] = command
        next_command = torch.empty_like(command)
        next_command[:-1] = torch.where(
            tensordict["next", "done"][:-1],
            command[:-1],
            command[1:],
        )
        next_command[-1] = command[-1]
        tensordict["command_state", "object_target_error_norm"] = object_target_error_norm
        tensordict["command_state", "cmd_object_vel_w"] = cmd_object_vel_w
        tensordict["command_state", "grasp_point_diff_w"] = grasp_point_diff_w
        tensordict["next", "command"] = next_command
        return tensordict


class LocoManipObjectScripted(_LocoManipObjectBase):
    """Scripted playback command using the ``SingleEEFLocoManip`` interface.

    Open-loop phase schedule per env (by ``episode_length_buf``):
    approach → grasp → lift → move.
    """

    def __init__(
        self,
        eef_body_name: str,
        gripper_joint_names: str,
        gripper_body_names: str,
        object_name: str = "object",
        grasp_height_range: Tuple[float, float] = (0.3, 0.5),
        standoff_distance: float = 0.65,
        standoff_linvel_gain: float = 2.0,
        standoff_yaw_gain: float = 1.0,
        speed_limit: float = 0.8,
        yaw_rate_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__(
            eef_body_name=eef_body_name,
            object_name=object_name,
            gripper_joint_names=gripper_joint_names,
            gripper_body_names=gripper_body_names,
        )
        self.grasp_height_range = grasp_height_range
        self.standoff_distance = standoff_distance
        self.standoff_linvel_gain = standoff_linvel_gain
        self.standoff_yaw_gain = standoff_yaw_gain
        self.speed_limit = speed_limit
        self.yaw_rate_range = yaw_rate_range

    @override
    def _initialize(self, env: "EnvBase") -> None:
        super()._initialize(env)

        body_ids, _ = self.asset.find_bodies(self.eef_body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected exactly one body matching {self.eef_body_name!r}, got {len(body_ids)}"
            )
        self.eef_body_idx = body_ids[0]

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
        move_offset[:, 0].uniform_(-2.0, 2.0)
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
        d = {
            "root_state_w": self.asset.data.root_state_w,
            "object_state_w": self.object.data.root_state_w,
            "grasp_point_w": self.grasp_point_w,
            "eef_state_w": self.asset.data.body_link_state_w[:, self.eef_body_idx],
            "eef_status": self.get_gripper_status(),
            "cmd_eef_status": self.cmd_eef_status,
        }
        self._state = TensorDict(d, [self.num_envs], device=self.device).clone()


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
    
    @override
    def get_state(self) -> TensorDict:
        return self._state


class object_distance_progress(RewardV2[LocoManipObject]):
    """Reward the reduction in object-to-target distance: ``prev_error - curr_error``."""

    @override
    def _initialize(self, env: "EnvBase") -> None:
        super()._initialize(env)
        self.prev_error_norm = torch.zeros(self.num_envs, 1, device=self.device)
        self.rew = torch.zeros(self.num_envs, 1, device=self.device)

    @override
    def reset(self, env_ids: torch.Tensor) -> None:
        self.prev_error_norm[env_ids] = self.command_manager.object_target_error_norm[
            env_ids
        ]
        self.rew[env_ids] = 0.0

    @override
    def update(self) -> None:
        curr_error = self.command_manager.object_target_error_norm
        self.rew = (self.prev_error_norm - curr_error) / self.env.step_dt
        self.prev_error_norm = curr_error.clone()

    @override
    def _compute(self) -> torch.Tensor:
        active = self.env.episode_length_buf > 1
        return self.rew.reshape(self.num_envs, 1), active.reshape(self.num_envs, 1)

    def relabel(self, tensordict: TensorDict) -> torch.Tensor:
        step_dt = tensordict["env_meta"]["step_dt"]
        T, N = tensordict.shape[:2]
        error_norm = tensordict["command_state", "object_target_error_norm"]
        rew = torch.cat(
            [
                torch.zeros(1, N, 1, device=tensordict.device),
                (error_norm[:-1] - error_norm[1:]) / step_dt,
            ],
            dim=0,
        )
        return rew.reshape(T, N, 1)


class object_pos_tracking(RewardV2[LocoManipObject]):
    """Exponential reward for tracking the commanded object target position."""

    def __init__(
        self,
        weight: float,
        enabled: bool = True,
        track_var: bool = False,
        pos_sigma: float = 0.25,
    ):
        super().__init__(weight, enabled, track_var)
        self.pos_sigma = pos_sigma

    @override
    def _compute(self) -> torch.Tensor:
        error = self.command_manager.object_target_error_norm
        rew = torch.exp(-error.square() / self.pos_sigma)
        return rew.reshape(self.num_envs, 1)

    def relabel(self, tensordict: TensorDict) -> torch.Tensor:
        error = tensordict["command_state", "object_target_error_norm"]
        return torch.exp(-error.square() / self.pos_sigma)


class object_vel_tracking(RewardV2[LocoManipObject]):
    """Exponential reward for tracking the commanded object linear velocity."""

    def __init__(
        self,
        weight: float,
        enabled: bool = True,
        track_var: bool = False,
        vel_sigma: float = 0.25,
    ):
        super().__init__(weight, enabled, track_var)
        self.vel_sigma = vel_sigma

    @override
    def _compute(self) -> torch.Tensor:
        diff = self.command_manager.cmd_object_vel_w - self.command_manager.object_vel_w
        error = diff.norm(dim=-1, keepdim=True)
        rew = torch.exp(-error.square() / self.vel_sigma)
        return rew.reshape(self.num_envs, 1)

    def relabel(self, tensordict: TensorDict) -> torch.Tensor:
        cmd_object_vel_w = tensordict["command_state", "cmd_object_vel_w"]
        object_vel_w = tensordict["command_state", "object_state_w"][..., 7:10]
        diff = cmd_object_vel_w - object_vel_w
        error = diff.norm(dim=-1, keepdim=True)
        return torch.exp(-error.square() / self.vel_sigma)


class object_grasp_pos(RewardV2[LocoManipObject]):
    """Reward reaching for the grasp point."""

    def __init__(
        self,
        weight: float,
        enabled: bool = True,
        track_var: bool = False,
        grasp_sigma: float = 0.25,
    ):
        super().__init__(weight, enabled, track_var)
        self.grasp_sigma = grasp_sigma

    @override
    def _compute(self) -> torch.Tensor:
        diff = self.command_manager.grasp_point_diff_w
        error = diff.norm(dim=-1, keepdim=True)
        return torch.exp(-error.square() / self.grasp_sigma)
    
    @override
    def relabel(self, tensordict: TensorDict) -> torch.Tensor:
        T, N = tensordict.shape[:2]
        diff = tensordict["command_state", "grasp_point_diff_w"]
        error = diff.norm(dim=-1, keepdim=True)
        rew = torch.exp(-error.square() / self.grasp_sigma)
        return rew.reshape(T, N, 1)


__all__ = [
    "LocoManipObject",
    "LocoManipObjectScripted",
    "object_distance_progress",
    "object_pos_tracking",
    "object_vel_tracking",
]
