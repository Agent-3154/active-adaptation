"""Locomotion + single end-effector position commands (scaffold)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from typing_extensions import override
from tensordict import TensorDict

from active_adaptation.utils.math import (
    clamp_norm,
    euler_rotate,
    quat_conjugate,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
    wrap_to_pi,
    yaw_quat,
    quat_from_euler_xyz
)
from active_adaptation.utils.symmetry import SymmetryTransform
from .base import CommandV2
from ..rewards.base import RewardV2

from dataclasses import dataclass, replace

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import EnvBase


@dataclass
class EEFCommandStruct:
    cmd_eef_pos_b: torch.Tensor
    cmd_eef_pos_w: torch.Tensor
    cmd_pos_world: torch.BoolTensor

    cmd_eef_rot_b: torch.Tensor
    cmd_eef_rot_w: torch.Tensor
    cmd_rot_world: torch.BoolTensor

    cmd_eef_forward_b: torch.Tensor
    cmd_eef_forward_w: torch.Tensor
    cmd_eef_upward_b: torch.Tensor
    cmd_eef_upward_w: torch.Tensor

    pos_diff_w: torch.Tensor
    pos_diff_b: torch.Tensor
    forward_diff_w: torch.Tensor
    forward_diff_b: torch.Tensor
    upward_diff_w: torch.Tensor
    upward_diff_b: torch.Tensor

    def sync(
        self,
        root_pos_w: torch.Tensor,
        root_yaw_quat: torch.Tensor,
        eef_pos_w: torch.Tensor,
        eef_quat_w: torch.Tensor,
    ) -> EEFCommandStruct:
        cmd_eef_pos_w, cmd_eef_pos_b = torch.cond(
            self.cmd_pos_world,
            self.world_from_body,
            self.body_from_world,
            (root_pos_w, root_yaw_quat, cmd_eef_pos_w, cmd_eef_pos_b)
        )
        
        pos_diff_w = self.cmd_eef_pos_w - eef_pos_w
        pos_diff_b = quat_rotate_inverse(
            root_yaw_quat,
            pos_diff_w
        )
        forward_diff_w = self.cmd_eef_forward_w - eef_quat_w
        forward_diff_b = quat_rotate_inverse(
            root_yaw_quat,
            forward_diff_w
        )
        upward_diff_w = self.cmd_eef_upward_w - eef_quat_w
        upward_diff_b = quat_rotate_inverse(
            root_yaw_quat,
            upward_diff_w
        )
        return replace(
            self,
            cmd_eef_pos_w=cmd_eef_pos_w,
            cmd_eef_pos_b=cmd_eef_pos_b,
            pos_diff_w=pos_diff_w,
            pos_diff_b=pos_diff_b,
            forward_diff_w=forward_diff_w,
            forward_diff_b=forward_diff_b,
            upward_diff_w=upward_diff_w,
            upward_diff_b=upward_diff_b,
        )
    
    @staticmethod
    def world_from_body(
        root_pos_w: torch.Tensor,
        root_yaw_quat: torch.Tensor,
        cmd_eef_pos_w: torch.Tensor,
        cmd_eef_pos_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cmd_eef_pos_w = (
            root_pos_w * torch.tensor([1., 1., 0.], device=root_pos_w.device)
            + quat_rotate(root_yaw_quat, cmd_eef_pos_b)
        )
        return cmd_eef_pos_w, cmd_eef_pos_b
    
    @staticmethod
    def body_from_world(
        root_pos_w: torch.Tensor,
        root_yaw_quat: torch.Tensor,
        cmd_eef_pos_w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cmd_eef_pos_b = quat_rotate_inverse(
            root_yaw_quat,
            cmd_eef_pos_w - root_pos_w * torch.tensor([1., 1., 0.], device=root_pos_w.device)
        )
        return cmd_eef_pos_w, cmd_eef_pos_b


class SingleEEFLocoManip(CommandV2):
    """Command vector: base velocity, yaw rate, EEF position, and EEF forward target.

    Dense layout (17D, body/yaw frame):
    ``[..., cmd_fwd_b(3), fwd_diff_b(3), cmd_closed(1), cmd_open(1)]``.
    Sparse layout (14D, body/yaw frame):
    ``[..., cmd_fwd_b(3), fwd_diff_b(3), cmd_closed(1), cmd_open(1)]``.

    The first two loco components are in the usual body horizontal frame (same as ``Twist``);
    ``eef_x``/``eef_y`` are **not** full body frame: they use the same **yaw-only**
    rotation as world ``(x,y)`` offsets from the root (pitch/roll of the base are
    ignored for the horizontal part). ``eef_z`` is **height above terrain**, not
    root-link ``z``: world target height is
    ``get_ground_height_at(query_xy) + eef_z``, with ``query_xy`` the horizontal target under
    the root. ``cmd_eef_forward_w`` is a world-frame unit vector specifying the commanded
    end-effector forward direction.
    """

    def __init__(
        self,
        eef_body_name: str,
        gripper_joint_names: str,
        workspace_range: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
        | None = None,
        workspace_profile: str | None = None,
        linvel_x_range: Tuple[float, float] = (-1.0, 1.0),
        linvel_y_range: Tuple[float, float] = (-1.0, 1.0),
        yaw_rate_range: Tuple[float, float] = (-torch.pi/2, torch.pi/2),
        world_goal_prob: float = 0.5,
        standoff_distance_range: Tuple[float, float] = (1.0, 2.0),
        standoff_linvel_gain_range: Tuple[float, float] = (1.0, 2.0),
        standoff_yaw_gain_range: Tuple[float, float] = (1.0, 2.0),
        resample_interval: int = 300,
        resample_prob: float = 0.75,
        cmd_eef_pos_clamp_range: float = -1.0,
    ) -> None:
        if workspace_range is None and workspace_profile is None:
            raise ValueError(
                "Either workspace_range or workspace_profile must be provided"
            )
        if workspace_range is not None and workspace_profile is not None:
            raise ValueError(
                "Only one of workspace_range or workspace_profile can be provided"
            )
        if not 0.0 <= world_goal_prob <= 1.0:
            raise ValueError("world_goal_prob must be in [0, 1]")

        self.eef_body_name = eef_body_name
        self.gripper_joint_names = gripper_joint_names
        self.workspace_range = workspace_range
        self.workspace_profile = workspace_profile
        self.linvel_x_range = linvel_x_range
        self.linvel_y_range = linvel_y_range
        self.yaw_rate_range = yaw_rate_range
        self.world_goal_prob = world_goal_prob
        self.standoff_distance_range = standoff_distance_range
        self.standoff_linvel_gain_range = standoff_linvel_gain_range
        self.standoff_yaw_gain_range = standoff_yaw_gain_range
        self.resample_interval = resample_interval
        self.resample_prob = resample_prob
        self.cmd_eef_pos_clamp_range = cmd_eef_pos_clamp_range

    @override
    def _initialize(self, env: "EnvBase") -> None:
        super()._initialize(env)
        body_ids, _ = self.asset.find_bodies(self.eef_body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected exactly one body matching {self.eef_body_name!r}, got {body_ids.numel()}"
            )
        self.eef_body_idx = body_ids[0]
        self.gripper_joint_ids, _ = self.asset.find_joints(self.gripper_joint_names)
        self.gripper_joint_ids = torch.tensor(self.gripper_joint_ids, device=self.device)
        limits = self.asset.data.soft_joint_pos_limits[0, self.gripper_joint_ids]
        self._gripper_max_open = limits.abs().amax(dim=-1).max().clamp_min(1e-6)

        with torch.device(self.device):
            if self.workspace_range is not None:
                lows = torch.tensor(
                    [self.workspace_range[i][0] for i in range(3)], dtype=torch.float32
                )
                highs = torch.tensor(
                    [self.workspace_range[i][1] for i in range(3)], dtype=torch.float32
                )
                self._eef_pos_low = lows.unsqueeze(0).expand(self.num_envs, -1).clone()
                self._eef_pos_high = highs.unsqueeze(0).expand(self.num_envs, -1).clone()
            self.standoff_linvel_gain = torch.zeros(self.num_envs, 1)
            self.standoff_yaw_gain = torch.zeros(self.num_envs, 1)

            self.cmd_linvel_b = torch.zeros(self.num_envs, 3)
            self.cmd_linvel_w = torch.zeros(self.num_envs, 3)
            self.cmd_yawvel_b = torch.zeros(self.num_envs, 1)
            # (x,y): horizontal offsets in yaw-aligned frame; z: height above ground at target xy.
            self.cmd_eef_pos_b = torch.zeros(self.num_envs, 3)
            self.cmd_eef_pos_w = torch.zeros(self.num_envs, 3)
            self.cmd_eef_pos_pd = torch.zeros(self.num_envs, 2)
            self.cmd_eef_pos_pd[:, 0].uniform_(5.0, 10.0)
            self.cmd_eef_pos_pd[:, 1] = 2.0 * self.cmd_eef_pos_pd[:, 0].sqrt()  # kd, ζ=1
            self.cmd_eef_vel_w = torch.zeros(self.num_envs, 3)  # computed by a PD controller
            self.cmd_eef_ref_vel_w = torch.zeros(self.num_envs, 3)
            self.eef_rot_w = torch.zeros(self.num_envs, 4)

            self.pos_diff_w = torch.zeros(self.num_envs, 3)
            self.pos_diff_b = torch.zeros(self.num_envs, 3)
            self.pos_error_norm2 = torch.zeros(self.num_envs, 1)
            self.pos_error_norm = torch.zeros(self.num_envs, 1)
            self.eef_pos_reached = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            self.eef_pos_reaching = torch.zeros(self.num_envs, 1, dtype=torch.bool)

            self.cmd_eef_rot_b = torch.zeros(self.num_envs, 4)
            self.cmd_eef_rot_w = torch.zeros(self.num_envs, 4)

            self.cmd_eef_forward_w = torch.zeros(self.num_envs, 3)
            self.cmd_eef_forward_b = torch.zeros(self.num_envs, 3)
            self.cmd_eef_upward_w = torch.zeros(self.num_envs, 3)
            self.cmd_eef_upward_b = torch.zeros(self.num_envs, 3)

            self.is_world_goal = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            self.local_env_ids = torch.empty(0, dtype=torch.long)
            self.world_env_ids = torch.empty(0, dtype=torch.long)
            # world goal eef position and velocity
            # the goal may move slowly
            self.world_eef_pos_w = torch.zeros(self.num_envs, 3)
            self.world_eef_vel_w = torch.zeros(self.num_envs, 3)

            # gripper closedness in [0, 1]: 0 = open, 1 = closed
            self.eef_status = torch.zeros(self.num_envs, 1)
            self.cmd_eef_status = torch.zeros(self.num_envs, 1, dtype=torch.long)
            # self.reaction_force

            self.standoff_pos_w = torch.zeros(self.num_envs, 3)
            self.standoff_yaw_w = torch.zeros(self.num_envs, 1)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            self.command_speed = torch.zeros(self.num_envs, 1)
            self.base_pos_error = torch.zeros(self.num_envs, 1)

            # payload applied at grasp point (force, unit: N)
            self.has_payload = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            self.payload_force_w = torch.zeros(self.num_envs, 3)
        
        self.init_eef_pos_b = quat_rotate_inverse(
            yaw_quat(self.asset.data.root_link_quat_w),
            self.eef_pos_w - self.asset.data.root_link_pos_w * torch.tensor([1., 1., 0.], device=self.device)
        )
        
        self.sync_state()

        self.marker = None
        self.standoff_marker = None
        if (
            self.env.backend == "isaac"
            and self.env.sim.has_gui()
        ):
            from active_adaptation.envs.backends.isaac import IsaacSceneAdapter

            self.scene: IsaacSceneAdapter = self.env.scene
            self.marker = self.scene.create_sphere_marker(
                "/Visuals/Command/target_eef_pos",
                color=(1.0, 0.4, 0.0),
                radius=0.03,
            )
            self.standoff_marker = self.scene.create_sphere_marker(
                "/Visuals/Command/standoff_pos",
                color=(0.0, 0.7, 1.0),
                radius=0.04,
            )
            self.eef_pose_marker = self.scene.create_frame_marker(
                "/Visuals/Command/target_eef_pose",
                scale=(0.1, 0.1, 0.1),
            )
    
    # @property
    # def cmd_eef_rot_w(self) -> torch.Tensor:
    #     return quat_mul(yaw_quat(self.asset.data.root_link_quat_w), self.cmd_eef_rot_b)
    
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

    def command(self, key: str = "dense") -> torch.Tensor:
        if key == "dense":
            cmd = torch.cat(
                [
                    self.cmd_linvel_b[:, :2], # [N, 2]
                    self.cmd_yawvel_b, # [N, 1]
                    self.cmd_eef_pos_b, # [N, 3]
                    self.pos_diff_b, # [N, 3]
                    self.cmd_eef_forward_b, # [N, 3]
                    self.forward_diff_b, # [N, 3]
                    self.cmd_eef_upward_b, # [N, 3]
                    self.upward_diff_b, # [N, 3]
                    self.cmd_eef_status.float(), # [N, 1]
                    (1 - self.cmd_eef_status.float()) # [N, 1]
                ],
                dim=-1,
            ) # [N, 23]
            assert cmd.shape == (self.num_envs, 23)
            return cmd
        elif key == "sparse":
            # align with LocoManipSparse's command
            return torch.cat([
                self.cmd_eef_pos_b, # [N, 3]
                self.pos_diff_b, # [N, 3]
                self.cmd_eef_forward_b, # [N, 3]
                self.forward_diff_b, # [N, 3]
                self.cmd_eef_upward_b, # [N, 3]
                self.upward_diff_b, # [N, 3]
                self.cmd_eef_status.float(), # [N, 1]
                (1 - self.cmd_eef_status.float()) # [N, 1]
            ], dim=-1)
        else:
            raise ValueError(f"Invalid key: {key}")

    @override
    def symmetry_transform(self, key: str = "dense"):
        if key == "dense":
            # flip y and yaw
            cmd_linvel_b = SymmetryTransform(perm=[0, 1], signs=[1, -1])
            cmd_yawvel_b = SymmetryTransform(perm=[0], signs=[-1])
            cmd_eef_pos_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            pos_diff_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            cmd_eef_forward_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            eef_forward_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            cmd_eef_upward_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            upward_diff_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            eef_status = SymmetryTransform(perm=[0, 1], signs=[1, 1])
            return SymmetryTransform.cat(
                [
                    cmd_linvel_b,
                    cmd_yawvel_b,
                    cmd_eef_pos_b,
                    pos_diff_b,
                    cmd_eef_forward_b,
                    eef_forward_b,
                    cmd_eef_upward_b,
                    upward_diff_b,
                    eef_status,
                ]
            )
        elif key == "sparse":
            # align with LocoManipSparse's sparse command
            cmd_eef_pos_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            pos_diff_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            cmd_eef_forward_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            forward_diff_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
            eef_status = SymmetryTransform(perm=[0, 1], signs=[1, 1])
            return SymmetryTransform.cat(
                [
                    cmd_eef_pos_b,
                    pos_diff_b,
                    cmd_eef_forward_b,
                    forward_diff_b,
                    cmd_eef_upward_b,
                    upward_diff_b,
                    eef_status,
                ]
            )
        else:
            raise ValueError(f"Invalid key: {key}")
    
    @override
    def pre_step(self, substep: int) -> None:
        self.asset._external_force_b[:, self.eef_body_idx] = quat_rotate_inverse(
            self.asset.data.body_link_quat_w[:, self.eef_body_idx],
            self.payload_force_w,
        )
        self.asset.has_external_wrench = True
    
    def get_gripper_status(self) -> torch.Tensor:
        """Return gripper closedness in ``[0, 1]`` (0=open, 1=closed)."""
        gripper_pos = self.asset.data.joint_pos[:, self.gripper_joint_ids]
        openness = (
            gripper_pos.abs().amax(dim=-1, keepdim=True) / self._gripper_max_open
        ).clamp(0.0, 1.0)
        return 1.0 - openness

    def sample_eef_status_commands(self, env_ids: torch.Tensor) -> None:
        self.cmd_eef_status[env_ids, 0] = torch.randint(
            0, 2, (len(env_ids),), device=self.device
        )
        has_payload = torch.rand(len(env_ids), 1, device=self.device) < 0.5
        payload_force_w = torch.zeros(len(env_ids), 3, device=self.device)
        payload_force_w[:, :2].uniform_(-10., 10.)
        payload_force_w[:, 2].uniform_(-20., 20.)
        self.payload_force_w[env_ids] = payload_force_w * has_payload
        self.has_payload[env_ids] = has_payload

    @staticmethod
    def _env_mask_prob(num_envs: int, prob: float, device: torch.device) -> torch.Tensor:
        return torch.rand(num_envs, device=device) < prob

    def _sample_local_eef_offsets(self, env_ids: torch.Tensor) -> torch.Tensor:
        if self.workspace_profile is not None:
            raise NotImplementedError(
                "workspace_profile sampling is not implemented; use workspace_range"
            )
        low = self._eef_pos_low[env_ids]
        high = self._eef_pos_high[env_ids]
        return torch.rand_like(low) * (high - low) + low

    def sample_loco_commands(self, env_ids: torch.Tensor) -> None: # env_ids is always non-empty
        # tensor[env_ids] is advanced indexing
        # so in-place operations like tensor[env_ids, 0].uniform_() have no effects
        new_cmd_linvel_b = torch.zeros(len(env_ids), 3, device=self.device)
        new_cmd_linvel_b[:, 0].uniform_(*self.linvel_x_range)
        new_cmd_linvel_b[:, 1].uniform_(*self.linvel_y_range)
        new_cmd_linvel_b[:, 2] = 0.0
        # reject speeds that are too small
        speed = new_cmd_linvel_b.norm(dim=-1)
        valid = speed > 0.1
        new_cmd_linvel_b[~valid] = 0.0
        new_cmd_yawvel_b = torch.zeros(len(env_ids), 1, device=self.device)
        new_cmd_yawvel_b[:, 0].uniform_(*self.yaw_rate_range)
        self.cmd_linvel_b[env_ids] = new_cmd_linvel_b
        self.cmd_yawvel_b[env_ids] = new_cmd_yawvel_b

    def sample_manip_commands(self, env_ids: torch.Tensor) -> None: # env_ids is always non-empty
        # in the body frame mode, always look body-frame forward
        cmd_eef_pos_b = self._sample_local_eef_offsets(env_ids)
        rpy = torch.zeros(len(env_ids), 3, device=self.device)
        rpy[:, 0].uniform_(-torch.pi / 2, torch.pi / 2)
        rpy[:, 1].uniform_(-torch.pi / 6, torch.pi / 6)
        rot_quat = quat_from_euler_xyz(rpy)
        self.cmd_eef_rot_b[env_ids] = rot_quat

        use_init = torch.rand(len(env_ids), 1, device=self.device) < 0.25
        self.cmd_eef_pos_b[env_ids] = torch.where(
            use_init,
            self.init_eef_pos_b[env_ids],
            cmd_eef_pos_b
        )
        self.cmd_eef_rot_b[env_ids] = torch.where(
            use_init,
            torch.tensor([[1., 0., 0., 0.]], device=self.device),
            rot_quat
        )

    def sample_world_goal_commands(self, env_ids: torch.Tensor) -> None:
        root_pos = self.asset.data.root_link_pos_w[env_ids]
        root_yaw_q = yaw_quat(self.asset.data.root_link_quat_w[env_ids])

        standoff_offset_b = torch.zeros(len(env_ids), 3, device=self.device)
        a = torch.rand(len(env_ids), device=self.device) * torch.pi * 2
        d = torch.empty(len(env_ids), device=self.device)
        d.uniform_(self.standoff_distance_range[0], self.standoff_distance_range[1])
        standoff_offset_b[:, 0] = d * torch.cos(a)
        standoff_offset_b[:, 1] = d * torch.sin(a)
        standoff_offset_w = quat_rotate(root_yaw_q, standoff_offset_b)
        standoff_pos_w = root_pos + standoff_offset_w
        standoff_pos_w[:, 2] = self.env.get_ground_height_at(standoff_pos_w)

        a = torch.rand(len(env_ids), device=self.device) * torch.pi * 2
        d = torch.empty(len(env_ids), device=self.device)
        d.uniform_(0.5, 0.7)
        eef_offset_w = torch.zeros(len(env_ids), 3, device=self.device)
        eef_offset_w[:, 0] = d * torch.cos(a)
        eef_offset_w[:, 1] = d * torch.sin(a)
        eef_offset_w[:, 2].uniform_(0.2, 0.9)
        world_eef_pos_w = standoff_pos_w + eef_offset_w
        world_eef_vel_w = torch.zeros(len(env_ids), 3, device=self.device)

        self.standoff_pos_w[env_ids] = standoff_pos_w
        yaw_gain = torch.empty(len(env_ids), 1, device=self.device)
        yaw_gain.uniform_(*self.standoff_yaw_gain_range)
        linvel_gain = torch.empty(len(env_ids), 1, device=self.device)
        linvel_gain.uniform_(*self.standoff_linvel_gain_range)
        self.standoff_yaw_gain[env_ids] = yaw_gain
        self.standoff_linvel_gain[env_ids] = linvel_gain

        self.world_eef_pos_w[env_ids] = world_eef_pos_w
        self.world_eef_vel_w[env_ids] = world_eef_vel_w
        
        delta_w = eef_offset_w.clone()
        delta_w[:, 2] -= root_pos[:, 2]
        horiz = torch.hypot(delta_w[:, 0], delta_w[:, 1])
        yaw = torch.atan2(delta_w[:, 1], delta_w[:, 0])
        yaw = yaw + torch.empty_like(yaw).uniform_(-torch.pi / 6, torch.pi / 6)
        pitch = torch.atan2(delta_w[:, 2], horiz)
        pitch = pitch + torch.empty_like(pitch).uniform_(-torch.pi / 6, torch.pi / 6)
        roll = torch.zeros_like(yaw)
        rpy_w = torch.stack([roll, pitch, yaw], dim=-1)

        self.standoff_yaw_w[env_ids, 0] = yaw
        self.cmd_eef_rot_w[env_ids] = quat_from_euler_xyz(rpy_w)

    def _split_command_strategy(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_world = int(len(env_ids) * self.world_goal_prob + 0.5)
        shuffled = env_ids[torch.randperm(len(env_ids), device=self.device)]
        world_env_ids = shuffled[:num_world]
        local_env_ids = shuffled[num_world:]
        return local_env_ids, world_env_ids

    def sample_commands(self, env_ids: torch.Tensor) -> None:
        local_env_ids, world_env_ids = self._split_command_strategy(env_ids)
        self.is_world_goal[env_ids] = False
        self.eef_pos_reached[env_ids] = False
        self.eef_pos_reaching[env_ids] = False
        if local_env_ids.numel() > 0:
            self.sample_loco_commands(local_env_ids)
            self.sample_manip_commands(local_env_ids)
        if world_env_ids.numel() > 0:
            self.is_world_goal[world_env_ids] = True
            self.sample_world_goal_commands(world_env_ids)
        self.sample_eef_status_commands(env_ids)
        keep_world = (self.world_env_ids[:, None] != env_ids[None, :]).all(dim=1)
        self.world_env_ids = torch.cat([self.world_env_ids[keep_world], world_env_ids])
        keep_local = (self.local_env_ids[:, None] != env_ids[None, :]).all(dim=1)
        self.local_env_ids = torch.cat([self.local_env_ids[keep_local], local_env_ids])

    def _sync_world_goal_envs(self, env_ids: torch.Tensor) -> None:
        # compute body-frame goals from world-frame goals
        root_pos = self.asset.data.root_link_pos_w[env_ids]
        yaw_q = self.root_yaw_quat[env_ids]

        if self.cmd_eef_pos_clamp_range > 0.0:
            cmd_eef_pos_w = (
                root_pos + 
                clamp_norm(self.world_eef_pos_w[env_ids] - root_pos, max=self.cmd_eef_pos_clamp_range)
            )
        else:
            cmd_eef_pos_w = self.world_eef_pos_w[env_ids]
        cmd_eef_pos_b = quat_rotate_inverse(
            yaw_q,
            cmd_eef_pos_w - root_pos * torch.tensor([1., 1., 0.], device=self.device)
        )
        self.cmd_eef_pos_w[env_ids] = cmd_eef_pos_w
        self.cmd_eef_pos_b[env_ids] = cmd_eef_pos_b
        self.cmd_eef_rot_b[env_ids] = quat_mul(quat_conjugate(yaw_q), self.cmd_eef_rot_w[env_ids])

        standoff_delta_w = self.standoff_pos_w[env_ids] - root_pos
        standoff_delta_w[:, 2] = 0.0
        standoff_delta_b = quat_rotate_inverse(yaw_q, standoff_delta_w)
        cmd_linvel_b = self.standoff_linvel_gain[env_ids] * standoff_delta_b
        cmd_linvel_b.clamp_(self.linvel_x_range[0], self.linvel_x_range[1])
        cmd_linvel_b.clamp_(self.linvel_y_range[0], self.linvel_y_range[1])
        cmd_linvel_b[:, 2] = 0.0
        self.cmd_linvel_b[env_ids] = cmd_linvel_b
        self.cmd_linvel_w[env_ids] = quat_rotate(yaw_q, cmd_linvel_b)
        self.base_pos_error[env_ids] = standoff_delta_b.norm(dim=-1, keepdim=True)

        yaw_error = wrap_to_pi(
            self.standoff_yaw_w[env_ids] - self.asset.data.heading_w[env_ids, None]
        )
        self.cmd_yawvel_b[env_ids] = (
            self.standoff_yaw_gain[env_ids] * yaw_error
        ).clamp(*self.yaw_rate_range)

    def _sync_world_frames(self) -> None:
        """Sync command tensors that are derived from the current root pose."""
        world_env_ids = self.world_env_ids
        if world_env_ids.numel() > 0:
            self._sync_world_goal_envs(world_env_ids)
        
        local_env_ids = self.local_env_ids
        if local_env_ids.numel() > 0:
            # compute world-frame quantities from body-frame
            self.cmd_linvel_w[local_env_ids] = quat_rotate(
                self.root_yaw_quat[local_env_ids],
                self.cmd_linvel_b[local_env_ids]
            )

            cmd_eef_pos_b = self.cmd_eef_pos_b[local_env_ids]
            self.cmd_eef_pos_w[local_env_ids] = (
                self.root_pos_w[local_env_ids] * torch.tensor([1., 1., 0.], device=self.device) 
                + quat_rotate(self.root_yaw_quat[local_env_ids], cmd_eef_pos_b)
            )
            self.cmd_eef_rot_w[local_env_ids] = quat_mul(
                self.root_yaw_quat[local_env_ids],
                self.cmd_eef_rot_b[local_env_ids]
            )
            self.base_pos_error[local_env_ids] = 0.0

        # always compute forward and upward in world frame
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
            self.cmd_eef_forward_w
        )
        self.cmd_eef_upward_b = quat_rotate_inverse(
            self.root_yaw_quat,
            self.cmd_eef_upward_w
        )

        self.command_speed = self.cmd_linvel_w.norm(dim=-1, keepdim=True)
        self.is_standing_env = (self.command_speed < 0.1)

    def _read_robot_state(self) -> None:
        self.root_pos_w = self.asset.data.root_link_pos_w
        self.root_yaw_quat = yaw_quat(self.asset.data.root_link_quat_w)
        self.eef_state_w = self.asset.data.body_link_state_w[:, self.eef_body_idx]

        forward_axis_b = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)
        upward_axis_b = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)
        self.eef_forward_w = quat_rotate(self.eef_quat_w, forward_axis_b)
        self.eef_forward_b = quat_rotate_inverse(self.root_yaw_quat, self.eef_forward_w)
        self.eef_upward_w = quat_rotate(self.eef_quat_w, upward_axis_b)
        self.eef_upward_b = quat_rotate_inverse(self.root_yaw_quat, self.eef_upward_w)
        self.eef_status = self.get_gripper_status()

    def _compute_tracking_errors(self) -> None:
        self.pos_diff_w = self.cmd_eef_pos_w - self.eef_pos_w
        self.pos_diff_b = quat_rotate_inverse(
            yaw_quat(self.asset.data.root_link_quat_w),
            self.pos_diff_w,
        )
        self.pos_error_norm2 = self.pos_diff_w.square().sum(dim=-1, keepdim=True)
        self.pos_error_norm = self.pos_error_norm2.sqrt()

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

    @override
    def reset(self, env_ids: torch.Tensor) -> None:
        self.sample_commands(env_ids)
        # self._sync_world_frames()
        self.base_pos_error[env_ids] = 0.0

    @override
    def sync_state(self) -> None:
        """Refresh tracking errors from post-physics robot state and current targets."""
        self._read_robot_state()
        self._sync_world_frames()
        self._compute_tracking_errors()
        d = {
            "is_world_goal": self.is_world_goal,
            "world_eef_pos_w": self.world_eef_pos_w,
            "cmd_eef_rot_w": self.cmd_eef_rot_w,
            "eef_state_w": self.eef_state_w,
            "eef_status": self.eef_status,
            "cmd_eef_status": self.cmd_eef_status,
            "root_state_w": self.asset.data.root_state_w,
            "base_pos_error": self.base_pos_error,
        }
        self._state = TensorDict(d, [self.num_envs], device=self.device).clone()

    @override
    def update(self) -> None:
        """Resample/advance targets for the next physics step, then refresh obs fields."""
        interval = (self.env.episode_length_buf - 20) % self.resample_interval == 0
        resample = interval & self._env_mask_prob(
            self.num_envs, self.resample_prob, self.device
        )
        env_ids = resample.nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() > 0:
            self.sample_commands(env_ids)
        
        world_env_ids = self.world_env_ids
        if world_env_ids.numel() == 0:
            return
        dpos = self.world_eef_vel_w[world_env_ids] * self.env.step_dt
        self.world_eef_pos_w[world_env_ids] += dpos
        self.standoff_pos_w[world_env_ids] += dpos

        self._sync_world_frames()
        self._compute_tracking_errors()

    @override
    def debug_draw(self) -> None:
        self.env.debug_draw.vector(
            self.asset.data.root_link_pos_w,
            self.cmd_linvel_w,
            color=(1.0, 1.0, 1.0, 1.0),
        )
        # self.env.debug_draw.vector(
        #     self.eef_pos_w,
        #     self.payload_force_w / 9.81,
        #     color=(0.0, 0.0, 1.0, 1.0),
        # )
        self.env.debug_draw.vector(
            self.eef_pos_w,
            self.cmd_eef_forward_w,
            color=(1.0, 0.0, 0.0, 1.0),
        )
        self.env.debug_draw.vector(
            self.eef_pos_w,
            self.cmd_eef_upward_w,
            color=(0.0, 0.0, 1.0, 1.0),
        )
        world_env_ids = self.world_env_ids
        
        if self.standoff_marker is not None and world_env_ids.numel() > 0:
            self.standoff_marker.visualize(self.standoff_pos_w[world_env_ids])
            self.marker.visualize(self.world_eef_pos_w[world_env_ids])
        
        self.eef_pose_marker.visualize(
            translations=self.cmd_eef_pos_w,
            orientations=self.cmd_eef_rot_w,
        )
    
    def get_state(self) -> TensorDict:
        return self._state


class eef_pos_tracking(RewardV2[SingleEEFLocoManip]):
    """Exponential position tracking with a small L1 penalty."""

    def __init__(self, weight: float, enabled: bool = True, track_var: bool = False):
        super().__init__(weight, enabled=enabled, track_var=track_var)
        self.sigma = 0.1

    @override
    def _compute(self) -> torch.Tensor:
        error_norm_sq = self.command_manager.pos_error_norm2
        error_norm = self.command_manager.pos_error_norm
        rew = torch.exp(-error_norm_sq / self.sigma) - 0.2 * error_norm
        return rew.reshape(self.num_envs, 1)


class eef_pos_error_l1(RewardV2[SingleEEFLocoManip]):
    """L1 end-effector position error (metric, not a shaped reward)."""

    @override
    def _compute(self) -> torch.Tensor:
        return self.command_manager.pos_error_norm.reshape(self.num_envs, 1)


class eef_pos_forward_tracking(RewardV2[SingleEEFLocoManip]):
    """Multiplicative reward of position and forward tracking."""

    def __init__(
        self,
        weight: float,
        enabled: bool = True,
        track_var: bool = False,
        base_pos_error_threshold: float = 1.0,
    ):
        super().__init__(weight, enabled=enabled, track_var=track_var)
        self.pos_sigma = 0.1
        self.rot_sigma = 0.4
        self.base_pos_error_threshold = base_pos_error_threshold
    
    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset = self.command_manager.asset
        self.eef_body_idx = self.command_manager.eef_body_idx
        self._base_height_rew = self.env.reward_groups["loco"]["base_height_exp"]
        self.update()
    
    @override
    def update(self) -> None:
        self.rew_pos_exp = torch.exp(-self.command_manager.pos_error_norm2 / self.pos_sigma)
        self.rew_pos_l1 = -0.2 * self.command_manager.pos_error_norm

        forward_diff = self.command_manager.forward_diff_w
        forward_error_norm2 = forward_diff.square().sum(dim=-1, keepdim=True)
        self.rew_forward = torch.exp(-forward_error_norm2 / self.rot_sigma)
        
        upward_diff = self.command_manager.upward_diff_w
        upward_error_norm2 = upward_diff.square().sum(dim=-1, keepdim=True)
        self.rew_upward = torch.exp(-upward_error_norm2 / self.rot_sigma)

        self._base_height_rew.modifier.mul_(self.rew_pos_exp)

    @override
    def _compute(self) -> torch.Tensor:
        active = self.command_manager.base_pos_error < self.base_pos_error_threshold
        rew = self.rew_pos_exp * self.rew_forward * self.rew_upward + self.rew_pos_l1
        return rew.reshape(self.num_envs, 1), active.reshape(self.num_envs, 1)
    
    def relabel(self, tensordict: TensorDict) -> torch.Tensor:
        T, N = tensordict.shape[:2]
        base_pos_error = tensordict["command_state", "base_pos_error"]
        pos_error_norm2 = tensordict["command_state", "pos_error_norm2"]
        pos_error_norm = tensordict["command_state", "pos_error_norm"]
        rew_pos = torch.exp(-pos_error_norm2 / self.pos_sigma)
        forward_diff = tensordict["command_state", "forward_diff_w"]
        forward_error_norm2 = forward_diff.square().sum(dim=-1, keepdim=True)
        rew_forward = torch.exp(-forward_error_norm2 / self.rot_sigma)
        upward_diff = tensordict["command_state", "upward_diff_w"]
        upward_error_norm2 = upward_diff.square().sum(dim=-1, keepdim=True)
        rew_upward = torch.exp(-upward_error_norm2 / self.rot_sigma)
        rew = rew_pos * rew_forward * rew_upward + -0.2 * pos_error_norm
        rew *= (base_pos_error < self.base_pos_error_threshold)
        return rew.reshape(T, N, 1)


class eef_pos_progress(RewardV2[SingleEEFLocoManip]):
    """Reward the reduction in EEF position error: ``prev_error - curr_error``."""
    
    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.prev_pos_error_norm = torch.zeros(self.num_envs, 1, device=self.device)
        self.rew = torch.zeros(self.num_envs, 1, device=self.device)

    @override
    def reset(self, env_ids: torch.Tensor) -> None:
        self.prev_pos_error_norm[env_ids] = self.command_manager.pos_error_norm[
            env_ids
        ]
        self.rew[env_ids] = 0.0

    @override
    def update(self) -> None:
        curr_error = self.command_manager.pos_error_norm
        self.rew = (self.prev_pos_error_norm - curr_error) / self.env.step_dt
        self.prev_pos_error_norm = curr_error.clone()

    @override
    def _compute(self) -> torch.Tensor:
        # the value may be incorrect at the first step
        active = (self.env.episode_length_buf > 1)
        return self.rew.reshape(self.num_envs, 1), active.reshape(self.num_envs, 1)
    
    def relabel(self, tensordict: TensorDict) -> torch.Tensor:
        # zero-pad the first step
        step_dt = tensordict["env_meta"]["step_dt"]
        T, N = tensordict.shape[:2]
        pos_error_norm = tensordict["command_state", "pos_error_norm"]
        rew = torch.cat([
            torch.zeros(1, N, 1, device=tensordict.device),
            (pos_error_norm[1:] - pos_error_norm[:-1]) / step_dt
        ], dim=0)
        return rew.reshape(T, N, 1)


class eef_pos_reaching(RewardV2[SingleEEFLocoManip]):
    """One-step reward for reaching the EEF position target."""

    @override
    def _compute(self) -> torch.Tensor:
        rew = torch.ones(self.num_envs, 1, device=self.device)
        return rew, self.command_manager.eef_pos_reaching


class eef_pos_reached(RewardV2[SingleEEFLocoManip]):
    """Reward for staying in an episode after the EEF position target is reached."""

    @override
    def _compute(self) -> torch.Tensor:
        rew = torch.ones(self.num_envs, 1, device=self.device)
        return rew, self.command_manager.eef_pos_reached


class eef_vel_tracking(RewardV2[SingleEEFLocoManip]):
    """Exponential reward for tracking commanded end-effector velocity."""

    def __init__(self, weight: float, enabled: bool = True, track_var: bool = False):
        super().__init__(weight, enabled=enabled, track_var=track_var)
        self.sigma = 0.25

    @override
    def _compute(self) -> torch.Tensor:
        diff_w = self.command_manager.cmd_eef_vel_w - self.command_manager.eef_vel_w
        error_l2 = diff_w.square().sum(dim=-1, keepdim=True)
        rew = torch.exp(-error_l2 / self.sigma)
        return rew.reshape(self.num_envs, 1)


class eef_forward_tracking(RewardV2[SingleEEFLocoManip]):
    """Track the commanded end-effector forward direction in world frame."""

    def __init__(
        self,
        weight: float,
        enabled: bool = True,
        track_var: bool = False,
        pos_error_threshold: float = 0.15,
    ):
        super().__init__(weight, enabled=enabled, track_var=track_var)
        self.pos_error_threshold = pos_error_threshold

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset = self.command_manager.asset
        self.eef_body_idx = self.command_manager.eef_body_idx

    @override
    def _compute(self) -> torch.Tensor:
        rew = (self.command_manager.eef_forward_w * self.command_manager.cmd_eef_forward_w).sum(
            dim=-1, keepdim=True
        )
        pos_error = (
            self.command_manager.cmd_eef_pos_w
            - self.asset.data.body_link_pos_w[:, self.eef_body_idx]
        ).norm(dim=-1, keepdim=True)
        active = pos_error < self.pos_error_threshold
        return rew.reshape(self.num_envs, 1), active.reshape(self.num_envs, 1)


class eef_up_tracking(RewardV2[SingleEEFLocoManip]):
    """Track a global EEF pitch target through the end-effector up direction."""

    def __init__(
        self,
        weight: float,
        enabled: bool = True,
        track_var: bool = False,
        pos_error_threshold: float = 0.15,
    ):
        super().__init__(weight, enabled=enabled, track_var=track_var)
        self.pos_error_threshold = pos_error_threshold

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset = self.command_manager.asset
        self.eef_body_idx = self.command_manager.eef_body_idx

    @override
    def _compute(self) -> torch.Tensor:
        rew = (
            self.command_manager.eef_upward_w * self.command_manager.cmd_eef_upward_w
        ).sum(dim=-1, keepdim=True)
        pos_error = (
            self.command_manager.cmd_eef_pos_w
            - self.asset.data.body_link_pos_w[:, self.eef_body_idx]
        ).norm(dim=-1, keepdim=True)
        active = pos_error < self.pos_error_threshold
        return rew.reshape(self.num_envs, 1), active.reshape(self.num_envs, 1)


class eef_angvel_penalty(RewardV2[SingleEEFLocoManip]):
    """Penalize end-effector angular velocity to reduce oscillation."""

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset = self.command_manager.asset
        self.eef_body_idx = self.command_manager.eef_body_idx

    @override
    def _compute(self) -> torch.Tensor:
        angvel = self.asset.data.body_link_ang_vel_w[:, self.eef_body_idx]
        rew = -angvel.square().sum(dim=-1, keepdim=True)
        return rew.reshape(self.num_envs, 1)
    
    def relabel(self, tensordict: TensorDict) -> TensorDict:
        T, N = tensordict.shape[:2]
        eef_state_w = tensordict["command_state", "eef_state_w"]
        eef_angvel_w = eef_state_w[..., 10:13]
        rew = -eef_angvel_w.square().sum(dim=-1, keepdim=True)
        return rew.reshape(T, N, 1)


class eef_grasp(RewardV2[SingleEEFLocoManip]):
    """Binary cross-entropy gripper reward on ``cmd_eef_status`` vs ``eef_status``."""

    @override
    def _compute(self) -> torch.Tensor:
        cmd = self.command_manager
        pred = cmd.eef_status.clamp(1e-6, 1.0 - 1e-6)
        target = cmd.cmd_eef_status.float()
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        return (1.0 - bce).reshape(self.num_envs, 1)
    
    def relabel(self, tensordict: TensorDict) -> TensorDict:
        T, N = tensordict.shape[:2]
        eef_status = tensordict["command_state", "eef_status"].clamp(1e-6, 1.0 - 1e-6)
        cmd_eef_status = tensordict["command_state", "cmd_eef_status"].float()
        bce = F.binary_cross_entropy(eef_status, cmd_eef_status, reduction="none")
        return (1.0 - bce).reshape(T, N, 1)


__all__ = ["SingleEEFLocoManip"]
