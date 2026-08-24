"""Task-space action terms (end-effector pose via differential IK)."""

from __future__ import annotations

import torch

from typing import TYPE_CHECKING, Mapping, Sequence
from typing_extensions import override

from tensordict import TensorDictBase

from active_adaptation.envs.utils import find_bodies, find_joints
from active_adaptation.utils.ik import DifferentialIKController
from active_adaptation.utils.string import resolve_matching_names_values
from active_adaptation.utils.math import (
    apply_delta_pose,
    matrix_from_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
)

from .base import ActionV2

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


class EndEffectorPose(ActionV2):
    """Absolute end-effector pose command in the articulation body frame.

    Policy actions are desired EE poses expressed in the robot root / body
    frame: ``(x, y, z)`` when ``orientation_weight == 0``, or
    ``(x, y, z, qw, qx, qy, qz)`` otherwise. Each physics substep solves one
    damped least-squares (DLS) differential IK step and writes joint position
    targets to the articulation PD controller.

    Optional **posture regularization** (``posture_weight > 0``) adds a
    null-space bias toward ``posture_target``. Unlisted joints use the asset
    default joint positions; override per joint via a name→value map (same
    pattern as randomization / ``JointReferenceModel`` scaling).

    When ``include_posture_target`` is true, the action vector appends one
    scalar per controlled joint (``posture_<joint_name>``) after the EE pose
    command. Each step updates the IK null-space target via
    :meth:`set_posture_target`.

    Relative (delta) commands are intentionally out of scope; use a separate
    ``EndEffectorPoseDelta`` term for that.
    """

    supported_backends = ("isaac", "mjlab")

    def __init__(
        self,
        joint_names: str | Sequence[str],
        body_name: str,
        *,
        damping: float = 0.05,
        max_dq: float = 0.5,
        position_weight: float = 1.0,
        orientation_weight: float = 1.0,
        posture_weight: float = 0.0,
        posture_target: Mapping[str, float] | None = None,
        include_posture_target: bool = False,
    ) -> None:
        super().__init__()
        self.joint_names_expr = joint_names
        self.body_name = body_name
        self.damping = float(damping)
        self.max_dq = float(max_dq)
        self.position_weight = float(position_weight)
        self.orientation_weight = float(orientation_weight)
        self.posture_weight = float(posture_weight)
        self.posture_target_cfg = posture_target
        self.include_posture_target = bool(include_posture_target)
        self.action_dim = 0
        self._ee_action_dim = 0

    @override
    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)

        joint_ids, self.joint_names = find_joints(self.asset, self.joint_names_expr)
        body_ids, body_names = find_bodies(self.asset, self.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected exactly one body matching {self.body_name!r}, "
                f"got {body_names}."
            )
        self.joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
        self.body_idx = int(body_ids[0])
        self.body_name_resolved = body_names[0]
        self.num_joints = len(joint_ids)

        posture_target = self._resolve_posture_target()
        self.controller = DifferentialIKController(
            self.num_envs,
            self.device,
            self.num_joints,
            damping=self.damping,
            max_dq=self.max_dq,
            position_weight=self.position_weight,
            orientation_weight=self.orientation_weight,
            posture_weight=self.posture_weight,
            posture_target=posture_target,
        )
        self._ee_action_dim = self.controller.action_dim
        self.action_dim = self._ee_action_dim
        if self.controller.track_orientation:
            ee_names = [
                "ee_x",
                "ee_y",
                "ee_z",
                "ee_qw",
                "ee_qx",
                "ee_qy",
                "ee_qz",
            ]
        else:
            ee_names = ["ee_x", "ee_y", "ee_z"]
        if self.include_posture_target:
            self.action_dim += self.num_joints
            self.names = ee_names + [
                f"posture_{name}" for name in self.joint_names
            ]
        else:
            self.names = ee_names

        with torch.device(self.device):
            self.raw_actions = torch.zeros(self.num_envs, self.action_dim)
            if self.controller.track_orientation:
                self.raw_actions[:, 3] = 1.0
            self._fill_posture_action_defaults()

        self._setup_jacobian_indexing()

    @property
    def _posture_action_dim(self) -> int:
        return self.num_joints if self.include_posture_target else 0

    def _fill_posture_action_defaults(self) -> None:
        if not self.include_posture_target:
            return
        start = self._ee_action_dim
        end = start + self._posture_action_dim
        self.raw_actions[:, start:end] = self.controller.posture_target

    def _apply_posture_action(self, action: torch.Tensor) -> None:
        if not self.include_posture_target:
            return
        start = self._ee_action_dim
        self.set_posture_target(action[:, start : start + self._posture_action_dim])

    def _resolve_posture_target(self) -> torch.Tensor:
        """Build ``(N, num_joints)`` posture target from defaults + optional overrides."""
        q_target = self.asset.data.default_joint_pos[:, self.joint_ids].clone()
        if self.posture_target_cfg is not None:
            _, _, overrides = resolve_matching_names_values(
                dict(self.posture_target_cfg),
                self.joint_names,
                preserve_order=True,
                strict=False,
            )
            for j, val in enumerate(overrides):
                if val is not None:
                    q_target[:, j] = val
        return q_target

    def set_posture_target(self, target: torch.Tensor) -> None:
        """Update runtime posture target ``q*`` (shape ``(N, num_joints)`` or ``(num_joints,)``)."""
        self.controller.set_posture_target(target)

    def _setup_jacobian_indexing(self) -> None:
        """Cache backend-specific Jacobian / body indices."""
        if self.env.backend == "isaac":
            # PhysX omits the root body column for fixed-base articulations and
            # prepends 6 floating-base DoFs for mobile bases.
            if self.asset.is_fixed_base:
                self._jacobi_body_idx = self.body_idx - 1
                self._jacobi_joint_ids = self.joint_ids
            else:
                self._jacobi_body_idx = self.body_idx
                self._jacobi_joint_ids = self.joint_ids + 6
            self._mj_joint_dof_ids = None
            self._mj_body_id = None
        elif self.env.backend == "mjlab":
            indexing = self.asset.indexing
            self._mj_joint_dof_ids = indexing.joint_v_adr[self.joint_ids]
            self._mj_body_id = int(indexing.body_ids[self.body_idx].item())
            self._jacobi_body_idx = None
            self._jacobi_joint_ids = None

            import warp as wp

            sim = self.env.sim._sim  # underlying mjlab Simulation
            nworld = self.num_envs
            nv = sim.mj_model.nv
            with wp.ScopedDevice(sim.wp_device):
                self._jacp_wp = wp.zeros((nworld, 3, nv), dtype=float)
                self._jacr_wp = wp.zeros((nworld, 3, nv), dtype=float)
                self._point_wp = wp.zeros(nworld, dtype=wp.vec3)
                self._body_wp = wp.zeros(nworld, dtype=wp.int32)
                self._body_wp.fill_(self._mj_body_id)
            self._jacp_torch = wp.to_torch(self._jacp_wp)
            self._jacr_torch = wp.to_torch(self._jacr_wp)
            self._point_torch = wp.to_torch(self._point_wp).view(nworld, 3)
        else:
            raise RuntimeError(
                f"EndEffectorPose does not support backend {self.env.backend!r}."
            )

    def __repr__(self) -> str:
        return (
            f"EndEffectorPose(body={self.body_name_resolved!r}, "
            f"joints={self.joint_names}, damping={self.damping}, "
            f"max_dq={self.max_dq}, position_weight={self.position_weight}, "
            f"orientation_weight={self.orientation_weight}, "
            f"posture_weight={self.posture_weight}, "
            f"include_posture_target={self.include_posture_target})"
        )

    @override
    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase) -> None:
        self.raw_actions[env_ids] = 0.0
        if self.controller.track_orientation:
            self.raw_actions[env_ids, 3] = 1.0
        self._fill_posture_action_defaults()
        self.controller.reset(env_ids)

    @override
    def process_action(self, action: torch.Tensor | None) -> None:
        if action is None:
            return
        self.raw_actions[:] = action
        ee_action = action[:, : self._ee_action_dim]
        self.controller.set_command(ee_action)
        self._apply_posture_action(action)

    @override
    def apply_action(self, substep: int) -> None:
        ee_pos_b = self._ee_pos_body()
        jacobian_pos_b = self._jacobian_pos_body()
        joint_pos = self.asset.data.joint_pos[:, self.joint_ids]
        if self.controller.track_orientation:
            joint_pos_des = self.controller.compute(
                ee_pos_b,
                jacobian_pos_b,
                joint_pos,
                ee_quat=self._ee_quat_body(),
                jacobian_rot=self._jacobian_rot_body(),
            )
        else:
            joint_pos_des = self.controller.compute(
                ee_pos_b, jacobian_pos_b, joint_pos
            )
        self.asset.set_joint_position_target(joint_pos_des, joint_ids=self.joint_ids)

    @override
    def symmetry_transform(self):
        raise NotImplementedError(
            "EndEffectorPose has no defined left/right symmetry transform."
        )

    @override
    def debug_draw(self) -> None:
        if not self.env.sim.has_gui():
            return
        root_pos = self.asset.data.root_link_pos_w
        root_quat = self.asset.data.root_link_quat_w
        des_pos_w = root_pos + quat_rotate(root_quat, self.controller.ee_pos_des)
        cur_pos_w = self.asset.data.body_link_pos_w[:, self.body_idx]
        self.env.scene.draw_point(des_pos_w, color=(0.1, 0.9, 0.2, 1.0), size=12.0)
        self.env.scene.draw_vector(
            cur_pos_w, des_pos_w - cur_pos_w, color=(0.2, 0.6, 1.0, 1.0), size=2.0
        )

    # ------------------------------------------------------------------
    # Pose / Jacobian (body frame)
    # ------------------------------------------------------------------

    def _ee_pos_body(self) -> torch.Tensor:
        """Current EE position in the articulation root / body frame."""
        ee_pos_w = self.asset.data.body_link_pos_w[:, self.body_idx]
        root_pos_w = self.asset.data.root_link_pos_w
        root_quat_w = self.asset.data.root_link_quat_w
        return quat_rotate_inverse(root_quat_w, ee_pos_w - root_pos_w)

    def _ee_quat_body(self) -> torch.Tensor:
        """Current EE orientation in the articulation root / body frame."""
        ee_quat_w = self.asset.data.body_link_quat_w[:, self.body_idx]
        root_quat_w = self.asset.data.root_link_quat_w
        return quat_mul(quat_conjugate(root_quat_w), ee_quat_w)

    def _jacobian_pos_body(self) -> torch.Tensor:
        """Geometric position Jacobian of the EE in the body frame.

        Shape ``(num_envs, 3, num_joints)``.
        """
        jacobian_w = self._jacobian_pos_world()
        root_quat_w = self.asset.data.root_link_quat_w
        rot_bw = matrix_from_quat(quat_conjugate(root_quat_w))
        return torch.bmm(rot_bw, jacobian_w)

    def _jacobian_rot_body(self) -> torch.Tensor:
        """Geometric rotational Jacobian of the EE in the body frame.

        Shape ``(num_envs, 3, num_joints)``.
        """
        jacobian_w = self._jacobian_rot_world()
        root_quat_w = self.asset.data.root_link_quat_w
        rot_bw = matrix_from_quat(quat_conjugate(root_quat_w))
        return torch.bmm(rot_bw, jacobian_w)

    def _jacobian_pos_world(self) -> torch.Tensor:
        """Geometric position Jacobian of the EE in the world frame."""
        if self.env.backend == "isaac":
            return self.asset.root_physx_view.get_jacobians()[
                :, self._jacobi_body_idx, :3, self._jacobi_joint_ids
            ]

        import mujoco_warp as mjwarp
        import warp as wp

        sim = self.env.sim._sim
        ee_pos_w = self.asset.data.body_link_pos_w[:, self.body_idx]
        self._point_torch[:] = ee_pos_w
        with wp.ScopedDevice(sim.wp_device):
            mjwarp.jac(
                sim.wp_model,
                sim.wp_data,
                self._jacp_wp,
                self._jacr_wp,
                self._point_wp,
                self._body_wp,
            )
        return self._jacp_torch[:, :, self._mj_joint_dof_ids]

    def _jacobian_rot_world(self) -> torch.Tensor:
        """Geometric rotational Jacobian of the EE in the world frame."""
        if self.env.backend == "isaac":
            return self.asset.root_physx_view.get_jacobians()[
                :, self._jacobi_body_idx, 3:6, self._jacobi_joint_ids
            ]

        self._jacobian_pos_world()
        return self._jacr_torch[:, :, self._mj_joint_dof_ids]


class EndEffectorPoseDelta(EndEffectorPose):
    """Relative end-effector pose command in the articulation body frame.

    Policy actions are deltas applied to the **current** EE pose each env step:

    - ``orientation_weight == 0``: ``(dx, dy, dz)`` position delta (meters)
    - ``orientation_weight > 0``: ``(dx, dy, dz, rx, ry, rz)`` with orientation
      delta in axis-angle form (radians)

    Deltas are scaled by ``delta_pos_scale`` / ``delta_ori_scale``, converted to
    an absolute target, then passed to the same DLS IK controller as
    :class:`EndEffectorPose`.
    """

    def __init__(
        self,
        joint_names: str | Sequence[str],
        body_name: str,
        *,
        delta_pos_scale: float = 1.0,
        delta_ori_scale: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(joint_names, body_name, **kwargs)
        self.delta_pos_scale = float(delta_pos_scale)
        self.delta_ori_scale = float(delta_ori_scale)

    @override
    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)
        if self.controller.track_orientation:
            self._ee_action_dim = 6
            ee_names = ["ee_dx", "ee_dy", "ee_dz", "ee_rx", "ee_ry", "ee_rz"]
        else:
            self._ee_action_dim = 3
            ee_names = ["ee_dx", "ee_dy", "ee_dz"]
        posture_names = (
            [f"posture_{name}" for name in self.joint_names]
            if self.include_posture_target
            else []
        )
        self.action_dim = self._ee_action_dim + self._posture_action_dim
        self.names = ee_names + posture_names
        with torch.device(self.device):
            self.raw_actions = torch.zeros(self.num_envs, self.action_dim)
            self._fill_posture_action_defaults()

    def __repr__(self) -> str:
        return (
            f"EndEffectorPoseDelta(body={self.body_name_resolved!r}, "
            f"joints={self.joint_names}, delta_pos_scale={self.delta_pos_scale}, "
            f"delta_ori_scale={self.delta_ori_scale})"
        )

    @override
    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase) -> None:
        self.raw_actions[env_ids] = 0.0
        self._fill_posture_action_defaults()
        self.controller.reset(env_ids)

    @override
    def process_action(self, action: torch.Tensor | None) -> None:
        if action is None:
            return
        self.raw_actions[:] = action
        ee_action = action[:, : self._ee_action_dim]
        ee_pos_b = self._ee_pos_body()
        if self.controller.track_orientation:
            ee_quat_b = self._ee_quat_body()
            delta = ee_action.clone()
            delta[:, :3] *= self.delta_pos_scale
            delta[:, 3:6] *= self.delta_ori_scale
            target_pos, target_quat = apply_delta_pose(ee_pos_b, ee_quat_b, delta)
            command = torch.cat([target_pos, target_quat], dim=-1)
        else:
            command = ee_pos_b + ee_action * self.delta_pos_scale
        self.controller.set_command(command)
        self._apply_posture_action(action)

    @override
    def symmetry_transform(self):
        raise NotImplementedError(
            "EndEffectorPoseDelta has no defined left/right symmetry transform."
        )


class EndEffectorPoseWithGripper(EndEffectorPose):
    """End-effector pose IK plus correlated gripper joint targets.

    The action vector concatenates the parent EE pose command (3D or 7D) with
    one or more gripper scalars mapped to joint position offsets via
    ``gripper_matrix`` (same convention as :class:`CorrelatedJointPosition`).

    Gripper scalars add ``action * gripper_action_scaling * matrix`` to the
    default joint positions. For the BlueROV parallel gripper, 0 maps to the
    lower (closed) limit and larger values move toward the upper (open) limit.
    """

    def __init__(
        self,
        joint_names: str | Sequence[str],
        body_name: str,
        gripper_joint_names: str | Sequence[str],
        *,
        gripper_matrix: list[float] | list[list[float]] | None = None,
        gripper_action_scaling: float = 1.0,
        **kwargs, # TODO: be explicit, never use **kwargs
    ) -> None:
        super().__init__(joint_names, body_name, **kwargs)
        self.gripper_joint_names_expr = gripper_joint_names
        self._gripper_matrix = gripper_matrix if gripper_matrix is not None else [1.0, 1.0]
        self.gripper_action_scaling = float(gripper_action_scaling)

    @override
    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)

        gripper_ids, self.gripper_joint_names = find_joints(
            self.asset, self.gripper_joint_names_expr
        )
        self.gripper_joint_ids = torch.tensor(
            gripper_ids, device=self.device, dtype=torch.long
        )

        coeffs = torch.tensor(self._gripper_matrix, dtype=torch.float32, device=self.device)
        if coeffs.ndim == 1:
            coeffs = coeffs.unsqueeze(-1)
        if coeffs.shape[0] != len(self.gripper_joint_names):
            raise ValueError(
                f"gripper_matrix rows ({coeffs.shape[0]}) must match number of "
                f"gripper joints ({len(self.gripper_joint_names)})"
            )
        self.gripper_matrix = coeffs
        self.gripper_action_dim = int(self.gripper_matrix.shape[1])

        gripper_names = [f"gripper_{i}" for i in range(self.gripper_action_dim)]
        self.action_dim = (
            self._ee_action_dim + self._posture_action_dim + self.gripper_action_dim
        )
        self.names = list(self.names) + gripper_names

        with torch.device(self.device):
            self.raw_actions = torch.zeros(self.num_envs, self.action_dim)
            self.gripper_action = torch.zeros(self.num_envs, self.gripper_action_dim)
            self.default_gripper_pos = self.asset.data.default_joint_pos[
                :, self.gripper_joint_ids
            ].clone()
            self._fill_posture_action_defaults()

    def __repr__(self) -> str:
        return (
            f"EndEffectorPoseWithGripper(body={self.body_name_resolved!r}, "
            f"joints={self.joint_names}, gripper_joints={self.gripper_joint_names}, "
            f"gripper_action_scaling={self.gripper_action_scaling})"
        )

    @property
    def _gripper_action_start(self) -> int:
        return self._ee_action_dim + self._posture_action_dim

    @override
    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase) -> None:
        super().reset(env_ids, tensordict)
        self.gripper_action[env_ids] = 0.0
        self.default_gripper_pos[env_ids] = self.asset.data.default_joint_pos[
            env_ids.unsqueeze(1), self.gripper_joint_ids
        ]

    @override
    def process_action(self, action: torch.Tensor | None) -> None:
        if action is None:
            return
        self.raw_actions[:] = action
        ee_action = action[:, : self._ee_action_dim]
        self.controller.set_command(ee_action)
        self._apply_posture_action(action)
        self.gripper_action[:] = action[:, self._gripper_action_start :]

    @override
    def apply_action(self, substep: int) -> None:
        super().apply_action(substep)
        joint_delta = (self.gripper_action @ self.gripper_matrix.T) * self.gripper_action_scaling
        jpos_target = self.default_gripper_pos + joint_delta
        self.asset.set_joint_position_target(
            jpos_target, joint_ids=self.gripper_joint_ids
        )

    @override
    def symmetry_transform(self):
        raise NotImplementedError(
            "EndEffectorPoseWithGripper has no defined left/right symmetry transform."
        )


__all__ = ["EndEffectorPose", "EndEffectorPoseDelta", "EndEffectorPoseWithGripper"]
