from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from typing_extensions import override
from tensordict import TensorDict

from active_adaptation.utils.math import (
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
    sample_quat_yaw,
    yaw_quat,
    quat_from_euler_xyz,
)
from active_adaptation.utils.symmetry import SymmetryTransform
from ..base import CommandV2

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import EnvBase


class LocoManipSparse(CommandV2):
    """Sparse loco-manip command: EEF position target only, no base velocity command.

    Sampling strategy:
    1. Sample a world-frame EEF target at each env origin with ``xy = [0, 0]`` and
       ``z`` in ``eef_z_range`` (terrain-relative height).
    2. Spawn the robot at a random position on a ring around the env origin, given
       by ``spawn_radius_range``.

    The policy-facing command is the heading-frame EEF target
    ``[eef_x, eef_y, eef_z]``, using the same convention as ``SingleEEFLocoManip``:
    horizontal components are yaw-aligned offsets from the root; ``eef_z`` is height
    above terrain at the target ``xy``.

    We expect this task to be harder to learn from scratch due to exploration difficulties:
    there is no signal for the base movement.
    """

    def __init__(
        self,
        eef_body_name: str,
        gripper_joint_names: str,
        eef_z_range: Tuple[float, float] = (0.2, 0.75),
        spawn_radius_range: Tuple[float, float] = (1.0, 3.0),
        resample_interval: int = 300,
        resample_prob: float = 0.75,
    ) -> None:
        self.eef_body_name = eef_body_name
        self.gripper_joint_names = gripper_joint_names
        self.eef_z_range = eef_z_range
        self.spawn_radius_range = spawn_radius_range
        self.resample_interval = resample_interval
        self.resample_prob = resample_prob

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
        self._gripper_max_open = limits[:, 1]

        with torch.device(self.device):
            self.cmd_eef_pos_b = torch.zeros(self.num_envs, 3)
            self.cmd_eef_pos_w = torch.zeros(self.num_envs, 3)
            self.eef_pos_w = torch.zeros(self.num_envs, 3)
            
            self.pos_diff_w = torch.zeros(self.num_envs, 3)
            self.pos_diff_b = torch.zeros(self.num_envs, 3)
            self.pos_error_norm2 = torch.zeros(self.num_envs, 1)
            self.pos_error_norm = torch.zeros(self.num_envs, 1)

            # orientation tracking
            self.cmd_eef_rot_w = torch.zeros(self.num_envs, 4)
            self.cmd_eef_rot_b = torch.zeros(self.num_envs, 4)

            self.eef_forward_w = torch.zeros(self.num_envs, 3)
            self.eef_forward_b = torch.zeros(self.num_envs, 3)
            self.cmd_eef_forward_w = torch.zeros(self.num_envs, 3)
            self.cmd_eef_forward_b = torch.zeros(self.num_envs, 3)
            self.cmd_eef_upward_w = torch.zeros(self.num_envs, 3)
            self.cmd_eef_upward_b = torch.zeros(self.num_envs, 3)

            # gripper closedness in [0, 1]: 0 = open, 1 = closed
            self.eef_status = torch.zeros(self.num_envs, 1)
            self.cmd_eef_status = torch.zeros(self.num_envs, 1, dtype=torch.long)

            self.world_eef_pos_w = torch.zeros(self.num_envs, 3)
            self.eef_pos_reaching = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            self.eef_pos_reached = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            self.eef_pos_reached_time = torch.zeros(self.num_envs, 1, dtype=torch.float)
            
            # payload applied at grasp point (force, unit: N)
            self.has_payload = torch.zeros(self.num_envs, 1, dtype=torch.bool)
            self.payload_force_w = torch.zeros(self.num_envs, 3)
            
            # we move the target in a continuous manner after it is reached
            self._scale_xyz = torch.zeros(self.num_envs, 3)
            self._scale_xyz[:, 0].uniform_(0.0, 0.6)
            self._scale_xyz[:, 1].uniform_(0.0, 0.3)
            self._scale_xyz[:, 2].uniform_(0.0, 0.1)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=torch.bool)

        self.marker = None
        if self.env.backend == "isaac" and self.env.sim.has_gui():
            from active_adaptation.envs.backends.isaac import IsaacSceneAdapter

            self.scene: IsaacSceneAdapter = self.env.scene
            self.eef_pose_marker = self.scene.create_frame_marker(
                "/Visuals/Command/target_eef_pose",
                scale=(0.1, 0.1, 0.1),
            )
        self.update()

    @property
    def command(self) -> torch.Tensor:
        # align with SingleEEFLocoManip's sparse command
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

    @override
    def symmetry_transform(self):
        cmd_eef_pos_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
        pos_diff_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
        cmd_eef_forward_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
        forward_diff_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
        cmd_eef_upward_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
        upward_diff_b = SymmetryTransform(perm=[0, 1, 2], signs=[1, -1, 1])
        eef_status = SymmetryTransform(perm=[0, 1], signs=[1, 1])
        return SymmetryTransform.cat([
            cmd_eef_pos_b,
            pos_diff_b,
            cmd_eef_forward_b,
            forward_diff_b,
            cmd_eef_upward_b,
            upward_diff_b,
            eef_status,
        ])
    
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

    @staticmethod
    def _env_mask_prob(num_envs: int, prob: float, device: torch.device) -> torch.Tensor:
        return torch.rand(num_envs, device=device) < prob

    def _sample_uniform(
        self, num_samples: int, value_range: Tuple[float, float]
    ) -> torch.Tensor:
        return (
            torch.rand(num_samples, device=self.device)
            * (value_range[1] - value_range[0])
            + value_range[0]
        )

    @override
    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        origins = self.env.scene.get_spawn_origins(env_ids)

        robot_init = self.init_root_state[env_ids].clone()
        default_z_offset = robot_init[:, 2].clone()

        angle = torch.rand(len(env_ids), device=self.device) * 2 * torch.pi
        radius = self._sample_uniform(len(env_ids), self.spawn_radius_range)
        robot_init[:, 0] = origins[:, 0] + radius * torch.cos(angle)
        robot_init[:, 1] = origins[:, 1] + radius * torch.sin(angle)
        robot_init[:, 2] = (
            self.env.get_ground_height_at(robot_init[:, :3]) + default_z_offset
        )
        robot_init[:, 3:7] = quat_mul(
            robot_init[:, 3:7],
            sample_quat_yaw(len(env_ids), device=self.device),
        )
        return robot_init

    def sample_commands(self, env_ids: torch.Tensor) -> None:
        origins = self.env.scene.env_origins[env_ids]
        z_offset = self._sample_uniform(len(env_ids), self.eef_z_range)
        target_w = origins.clone()
        target_w[:, 2] = self.env.get_ground_height_at(origins) + z_offset
        self.world_eef_pos_w[env_ids] = target_w

        rpy_w = torch.zeros(len(env_ids), 3, device=self.device)
        rpy_w[:, 0].uniform_(-torch.pi / 2, torch.pi / 2)
        rpy_w[:, 1].uniform_(-torch.pi / 6, torch.pi / 6)
        self.cmd_eef_rot_w[env_ids] = quat_from_euler_xyz(rpy_w)

        # self.eef_pos_reaching[env_ids] = False
        # self.eef_pos_reached[env_ids] = False
        # self.eef_pos_reached_time[env_ids] = 0.0

    @override
    def reset(self, env_ids: torch.Tensor) -> None:
        self.sample_commands(env_ids)
        # Not need to update states here because we do not run FK after reset,
        # so the body states are not updated yet.
        # Running FK at each reset is expensive.
        # self._sync_command()

    @override
    def update(self) -> None:
        # update common terms
        self.root_pos_w = self.asset.data.root_link_pos_w
        self.root_yaw_quat = yaw_quat(self.asset.data.root_link_quat_w)
        self.eef_pos_w = self.asset.data.body_link_pos_w[:, self.eef_body_idx]
        self.eef_quat_w = self.asset.data.body_link_quat_w[:, self.eef_body_idx]
        forward_axis_b = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)
        upward_axis_b = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)
        self.eef_forward_w = quat_rotate(self.eef_quat_w, forward_axis_b)
        self.eef_forward_b = quat_rotate_inverse(self.root_yaw_quat, self.eef_forward_w)
        self.eef_upward_w = quat_rotate(self.eef_quat_w, upward_axis_b)
        self.eef_upward_b = quat_rotate_inverse(self.root_yaw_quat, self.eef_upward_w)
        self.eef_status = self.get_gripper_status()

        interval = (self.env.episode_length_buf - 20) % self.resample_interval == 0
        resample = (
            interval
            & self._env_mask_prob(self.num_envs, self.resample_prob, self.device)
            & self.eef_pos_reached.squeeze(1) # do not resample if not reached yet
        )
        env_ids = resample.nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() > 0:
            self.sample_commands(env_ids)
        
        self.cmd_eef_pos_w = self.world_eef_pos_w.clone()
        self.cmd_eef_pos_b = quat_rotate_inverse(
            self.root_yaw_quat,
            self.world_eef_pos_w - self.root_pos_w * torch.tensor([1.0, 1.0, 0.0], device=self.device),
        )
        
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
            self.forward_diff_w
        )
        self.upward_diff_w = self.cmd_eef_upward_w - self.eef_upward_w
        self.upward_diff_b = quat_rotate_inverse(
            self.root_yaw_quat,
            self.upward_diff_w
        )

    @override
    def debug_draw(self) -> None:
        self.env.debug_draw.vector(
            self.eef_pos_w,
            self.cmd_eef_pos_w - self.eef_pos_w,
            color=(0.0, 0.0, 1.0, 1.0),
        )
        self.eef_pose_marker.visualize(
            translations=self.cmd_eef_pos_w,
            orientations=self.cmd_eef_rot_w,
        )
        
    @staticmethod
    def relabel_command(tensordict: TensorDict) -> TensorDict:
        """Compute the necessary states and commands from a rollout.
        This is used to relabel the command from `SingleEEFLocoManip` to `LocoManipSparse`.
        """
        device = tensordict.device
        assert tensordict["is_world_goal"].all()
        root_pos_w = tensordict["root_state_w"][..., :3]
        root_quat_w = tensordict["root_state_w"][..., 3:7]
        root_yaw_quat = yaw_quat(root_quat_w)
        cmd_eef_pos_w = tensordict["world_eef_pos_w"]
        cmd_eef_pos_b = quat_rotate_inverse(
            root_yaw_quat,
            cmd_eef_pos_w - root_pos_w * torch.tensor([1.0, 1.0, 0.0], device=device)
        )
        cmd_eef_rot_w = tensordict["cmd_eef_rot_w"]
        cmd_eef_forward_w = quat_rotate(cmd_eef_rot_w, torch.tensor([[1.0, 0.0, 0.0]], device=device))
        cmd_eef_forward_b = quat_rotate_inverse(root_yaw_quat, cmd_eef_forward_w)
        cmd_eef_upward_w = quat_rotate(cmd_eef_rot_w, torch.tensor([[0.0, 0.0, 1.0]], device=device))
        cmd_eef_upward_b = quat_rotate_inverse(root_yaw_quat, cmd_eef_upward_w)

        eef_pos_w = tensordict["eef_pos_w"]
        eef_quat_w = tensordict["eef_quat_w"]
        pos_diff_w = cmd_eef_pos_w - eef_pos_w
        pos_diff_b = quat_rotate_inverse(root_yaw_quat, pos_diff_w)
        pos_error_norm2 = pos_diff_w.square().sum(dim=-1, keepdim=True)
        pos_error_norm = pos_error_norm2.sqrt()
        eef_forward_w = quat_rotate(eef_quat_w, torch.tensor([[1.0, 0.0, 0.0]], device=device))
        eef_upward_w = quat_rotate(eef_quat_w, torch.tensor([[0.0, 0.0, 1.0]], device=device))
        forward_diff_b = quat_rotate_inverse(root_yaw_quat, cmd_eef_forward_w - eef_forward_w)
        upward_diff_b = quat_rotate_inverse(root_yaw_quat, cmd_eef_upward_w - eef_upward_w)
        
        command_sparse = torch.cat([
            cmd_eef_pos_b,
            pos_diff_b,
            cmd_eef_forward_b,
            forward_diff_b,
            cmd_eef_upward_b,
            upward_diff_b,
            tensordict["eef_status"]
        ], dim=-1)

        tensordict["pos_error_norm2"] = pos_error_norm2
        tensordict["pos_error_norm"] = pos_error_norm
        tensordict["command"] = command_sparse
        return tensordict


__all__ = ["LocoManipSparse"]
