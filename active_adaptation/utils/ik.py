"""Differential inverse-kinematics helpers (backend-agnostic math)."""

from __future__ import annotations

import torch


def damped_least_squares(
    jacobian: torch.Tensor,
    dx: torch.Tensor,
    damping: float,
    max_dq: float | None = None,
) -> torch.Tensor:
    """Solve joint-space damped least squares: ``(JᵀJ + λ²I) dq = Jᵀ dx``.

    Args:
        jacobian: Geometric Jacobian. Shape ``(N, task_dim, num_joints)``.
        dx: Task-space residual. Shape ``(N, task_dim)``.
        damping: Damping coefficient ``λ``.
        max_dq: Optional per-component clamp on ``dq`` (rad).

    Returns:
        Joint displacement ``dq`` of shape ``(N, num_joints)``.
    """
    lam = max(float(damping), 1e-6)
    # (Jᵀ J + λ² I) dq = Jᵀ dx  — n×n system (friendly for future null-space rows).
    jtj = torch.einsum("bti,btj->bij", jacobian, jacobian)
    jtdx = torch.einsum("bti,bt->bi", jacobian, dx)
    jtj.diagonal(dim1=-2, dim2=-1).add_(lam * lam)
    dq = torch.linalg.solve(jtj, jtdx)
    if max_dq is not None:
        dq = dq.clamp(-max_dq, max_dq)
    return dq


class DifferentialIKController:
    """Position-only differential IK via damped least squares.

    Commands and Jacobians are expected in the **articulation root / body
    frame**. Absolute position targets only; relative (delta) commands belong
    in a separate action term.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        damping: float = 0.05,
        max_dq: float = 0.5,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.damping = float(damping)
        self.max_dq = float(max_dq)
        self.ee_pos_des = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def action_dim(self) -> int:
        """Absolute EE position ``(x, y, z)`` in the body frame."""
        return 3

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.ee_pos_des[env_ids] = 0.0

    def set_command(self, command: torch.Tensor) -> None:
        """Set absolute end-effector position target in the body frame.

        Args:
            command: Desired EE position. Shape ``(N, 3)``.
        """
        self.ee_pos_des[:] = command

    def compute(
        self,
        ee_pos: torch.Tensor,
        jacobian: torch.Tensor,
        joint_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Compute joint position targets for the current desired EE position.

        Args:
            ee_pos: Current EE position in the body frame. Shape ``(N, 3)``.
            jacobian: Position Jacobian in the body frame.
                Shape ``(N, 3, num_joints)``.
            joint_pos: Current controlled joint positions.
                Shape ``(N, num_joints)``.

        Returns:
            Desired joint positions. Shape ``(N, num_joints)``.
        """
        pos_error = self.ee_pos_des - ee_pos
        dq = damped_least_squares(
            jacobian, pos_error, self.damping, max_dq=self.max_dq
        )
        return joint_pos + dq
