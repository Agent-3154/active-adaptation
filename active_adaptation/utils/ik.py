"""Differential inverse-kinematics helpers (backend-agnostic math)."""

from __future__ import annotations

import torch

from active_adaptation.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul


def _add_posture_regularization(
    jtj: torch.Tensor,
    jtdx: torch.Tensor,
    joint_pos: torch.Tensor,
    posture_target: torch.Tensor,
    posture_weight: float,
) -> None:
    """Add null-space posture bias: ``w² (q* − q)`` to the DLS normal equations."""
    w2 = float(posture_weight) ** 2
    jtj.diagonal(dim1=-2, dim2=-1).add_(w2)
    jtdx.add_(w2 * (posture_target - joint_pos))


def damped_least_squares(
    jacobian: torch.Tensor,
    dx: torch.Tensor,
    damping: float,
    max_dq: float | None = None,
    *,
    joint_pos: torch.Tensor | None = None,
    posture_target: torch.Tensor | None = None,
    posture_weight: float = 0.0,
) -> torch.Tensor:
    """Solve joint-space damped least squares: ``(JᵀJ + λ²I) dq = Jᵀ dx``.

    Args:
        jacobian: Geometric Jacobian. Shape ``(N, task_dim, num_joints)``.
        dx: Task-space residual. Shape ``(N, task_dim)``.
        damping: Damping coefficient ``λ``.
        max_dq: Optional per-component clamp on ``dq`` (rad).
        joint_pos: Current joint positions for posture regularization.
        posture_target: Preferred joint posture ``q*``. Shape ``(N, num_joints)``.
        posture_weight: Weight on ``(q* − q)`` null-space term (0 = disabled).

    Returns:
        Joint displacement ``dq`` of shape ``(N, num_joints)``.
    """
    lam = max(float(damping), 1e-6)
    jtj = torch.einsum("bti,btj->bij", jacobian, jacobian)
    jtdx = torch.einsum("bti,bt->bi", jacobian, dx)
    if posture_weight > 0.0:
        if joint_pos is None or posture_target is None:
            raise ValueError(
                "Posture regularization requires `joint_pos` and `posture_target`."
            )
        _add_posture_regularization(jtj, jtdx, joint_pos, posture_target, posture_weight)
    jtj.diagonal(dim1=-2, dim2=-1).add_(lam * lam)
    dq = torch.linalg.solve(jtj, jtdx)
    if max_dq is not None:
        dq = dq.clamp(-max_dq, max_dq)
    return dq


def damped_least_squares_pose(
    jacobian_pos: torch.Tensor,
    jacobian_rot: torch.Tensor,
    pos_error: torch.Tensor,
    rot_error: torch.Tensor,
    damping: float,
    *,
    position_weight: float = 1.0,
    orientation_weight: float = 1.0,
    max_dq: float | None = None,
    joint_pos: torch.Tensor | None = None,
    posture_target: torch.Tensor | None = None,
    posture_weight: float = 0.0,
) -> torch.Tensor:
    """Solve weighted 6D pose IK in joint space via damped least squares."""
    lam = max(float(damping), 1e-6)
    wp2 = float(position_weight) ** 2
    wo2 = float(orientation_weight) ** 2
    jtj = wp2 * torch.einsum("bti,btj->bij", jacobian_pos, jacobian_pos)
    jtj.add_(wo2 * torch.einsum("bti,btj->bij", jacobian_rot, jacobian_rot))
    jtdx = wp2 * torch.einsum("bti,bt->bi", jacobian_pos, pos_error)
    jtdx.add_(wo2 * torch.einsum("bti,bt->bi", jacobian_rot, rot_error))
    if posture_weight > 0.0:
        if joint_pos is None or posture_target is None:
            raise ValueError(
                "Posture regularization requires `joint_pos` and `posture_target`."
            )
        _add_posture_regularization(jtj, jtdx, joint_pos, posture_target, posture_weight)
    jtj.diagonal(dim1=-2, dim2=-1).add_(lam * lam)
    dq = torch.linalg.solve(jtj, jtdx)
    if max_dq is not None:
        dq = dq.clamp(-max_dq, max_dq)
    return dq


class DifferentialIKController:
    """Differential IK via damped least squares.

    Commands and Jacobians are expected in the **articulation root / body
    frame**. Supports absolute position-only (3D) or pose (7D: position +
    quaternion ``wxyz``) commands.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        num_joints: int,
        *,
        damping: float = 0.05,
        max_dq: float = 0.5,
        position_weight: float = 1.0,
        orientation_weight: float = 1.0,
        posture_weight: float = 0.0,
        posture_target: torch.Tensor | None = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.num_joints = int(num_joints)
        self.damping = float(damping)
        self.max_dq = float(max_dq)
        self.position_weight = float(position_weight)
        self.orientation_weight = float(orientation_weight)
        self.posture_weight = float(posture_weight)
        self.ee_pos_des = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_quat_des = torch.zeros(self.num_envs, 4, device=self.device)
        self.ee_quat_des[:, 0] = 1.0
        self.posture_target = torch.zeros(
            self.num_envs, self.num_joints, device=self.device
        )
        if posture_target is not None:
            self.set_posture_target(posture_target)

    @property
    def track_orientation(self) -> bool:
        return self.orientation_weight > 0.0

    @property
    def action_dim(self) -> int:
        """Absolute EE command dimension in the body frame."""
        return 7 if self.track_orientation else 3

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.ee_pos_des[env_ids] = 0.0
        self.ee_quat_des[env_ids] = 0.0
        self.ee_quat_des[env_ids, 0] = 1.0

    def set_command(self, command: torch.Tensor) -> None:
        """Set absolute end-effector target in the body frame.

        Args:
            command: ``(N, 3)`` position or ``(N, 7)`` position + quaternion
                ``(w, x, y, z)`` when orientation tracking is enabled.
        """
        self.ee_pos_des[:] = command[:, :3]
        if not self.track_orientation:
            return
        if command.shape[-1] != 7:
            raise ValueError(
                f"Expected orientation command with shape (N, 7), got {tuple(command.shape)}."
            )
        quat = command[:, 3:7]
        self.ee_quat_des[:] = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def set_posture_target(self, target: torch.Tensor) -> None:
        """Set preferred joint posture ``q*`` for null-space regularization.

        Args:
            target: Shape ``(N, num_joints)`` or ``(num_joints,)`` (broadcast).
        """
        if target.ndim == 1:
            self.posture_target[:] = target.unsqueeze(0)
        else:
            self.posture_target[:] = target

    def compute(
        self,
        ee_pos: torch.Tensor,
        jacobian: torch.Tensor,
        joint_pos: torch.Tensor,
        ee_quat: torch.Tensor | None = None,
        jacobian_rot: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute joint position targets for the current desired EE pose.

        Args:
            ee_pos: Current EE position in the body frame. Shape ``(N, 3)``.
            jacobian: Position Jacobian in the body frame.
                Shape ``(N, 3, num_joints)``.
            joint_pos: Current controlled joint positions.
                Shape ``(N, num_joints)``.
            ee_quat: Current EE orientation in the body frame ``(w, x, y, z)``.
                Required when orientation tracking is enabled.
            jacobian_rot: Rotational Jacobian in the body frame.
                Shape ``(N, 3, num_joints)``. Required with ``ee_quat``.

        Returns:
            Desired joint positions. Shape ``(N, num_joints)``.
        """
        pos_error = self.ee_pos_des - ee_pos
        posture_kw = {}
        if self.posture_weight > 0.0:
            posture_kw = {
                "joint_pos": joint_pos,
                "posture_target": self.posture_target,
                "posture_weight": self.posture_weight,
            }
        if not self.track_orientation:
            dq = damped_least_squares(
                jacobian,
                pos_error,
                self.damping,
                max_dq=self.max_dq,
                **posture_kw,
            )
            return joint_pos + dq

        if ee_quat is None or jacobian_rot is None:
            raise ValueError(
                "Orientation tracking requires `ee_quat` and `jacobian_rot`."
            )
        quat_error = quat_mul(self.ee_quat_des, quat_conjugate(ee_quat))
        rot_error = axis_angle_from_quat(quat_error)
        dq = damped_least_squares_pose(
            jacobian,
            jacobian_rot,
            pos_error,
            rot_error,
            self.damping,
            position_weight=self.position_weight,
            orientation_weight=self.orientation_weight,
            max_dq=self.max_dq,
            **posture_kw,
        )
        return joint_pos + dq
