from __future__ import annotations

import torch
import torch.nn as nn

from typing import Dict, Optional, Tuple
from typing_extensions import override

try:
    import isaaclab.utils.string as string_utils
except ModuleNotFoundError:
    from mjlab.utils.lab_api import string as string_utils

from active_adaptation.utils.symmetry import SymmetryTransform, joint_space_symmetry

from .base import Action


class SoftBoundTracker(nn.Module):
    """
    Streaming soft bounds via online quantile (pinball) updates — fixed memory.

    ``lower`` / ``upper`` are buffers with the given ``shape`` (e.g. ``D`` or
    ``(D,)`` for per-feature bounds). Observations ``x`` must end with that
    shape; leading dimensions are i.i.d. samples. Initialized to zero; call
    ``reset()`` to zero again.

    Args:
        shape: Bound tensor shape; an ``int`` is treated as ``(int,)``.
    """

    tau: float
    lr: float

    def __init__(
        self,
        shape: torch.Size | Tuple[int, ...],
        *,
        tau: float = 0.9,
        lr: float = 0.05,
    ):
        super().__init__()
        if not 0.0 < tau < 1.0:
            raise ValueError(f"tau must be in (0, 1), got {tau}")
        self.p_lo = 1.0 - tau
        self.p_hi = tau
        self.lr = float(lr)
        sz = torch.Size(shape)
        self.register_buffer("lower", torch.zeros(sz),)
        self.register_buffer("upper", torch.zeros(sz),)

    def extra_repr(self) -> str:
        return f"shape={tuple(self.lower.shape)}, tau={self.tau}, lr={self.lr}"

    def reset(self) -> None:
        self.lower.zero_()
        self.upper.zero_()

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Incorporate a minibatch; refine ``lower`` / ``upper`` in place."""
        dt = self.lower.dtype
        ind_lo = (x < self.lower).to(dt)
        ind_hi = (x < self.upper).to(dt)
        g_lo = (self.p_lo - ind_lo).mean(dim=0)
        g_hi = (self.p_hi - ind_hi).mean(dim=0)
        lo = self.lower + self.lr * g_lo
        hi = self.upper + self.lr * g_hi
        self.lower.copy_(torch.minimum(lo, hi))
        self.upper.copy_(torch.maximum(lo, hi))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.update(x)
        return self.lower, self.upper


class _DelayedJointAction(Action):
    def __init__(
        self,
        env,
        action_scaling: Dict[str, float] = 0.5,
        max_delay: int = 2,
        alpha_range: Tuple[float, float] = (0.5, 1.0),
        track_pos_target_bounds: bool = False,
        track_vel_target_bounds: bool = False,
    ):
        super().__init__(env)
        self.track_pos_target_bounds = track_pos_target_bounds
        self.track_vel_target_bounds = track_vel_target_bounds

        if isinstance(action_scaling, float):
            action_scaling = {".*": float(action_scaling)}
        
        _, self.joint_names, scaling = string_utils.resolve_matching_names_values(
            dict(action_scaling), self.asset.cfg.joint_names_simulation
        )
        self.joint_ids = torch.tensor(
            [self.asset.joint_names.index(name) for name in self.joint_names],
            device=self.device,
        )

        self.action_scaling = torch.tensor(scaling, device=self.device)
        self.max_delay = max_delay
        self.alpha_range = tuple(alpha_range)
        self.decimation = int(self.env.step_dt / self.env.physics_dt)

        with torch.device(self.device):
            self.action_buf = torch.zeros(self.num_envs, 4, self.action_dim)
            self.action_queue = torch.zeros(
                self.num_envs,
                self.max_delay + self.decimation,
                self.action_dim,
            )
            self.applied_action = torch.zeros(self.num_envs, self.action_dim)
            self.alpha = torch.ones(self.num_envs, 1)
            self.delay = torch.zeros(self.num_envs, 1, dtype=torch.int64)
        
        if self.track_pos_target_bounds:
            self.pos_target_bound_tracker = SoftBoundTracker(
                shape=(self.action_dim,), tau=0.9
            ).to(self.device)
        if self.track_vel_target_bounds:
            self.vel_target_bound_tracker = SoftBoundTracker(
                shape=(self.action_dim,), tau=0.9
            ).to(self.device)

    def diagnostics(self) -> dict:
        d = {}
        if self.track_pos_target_bounds:
            for i, jname in enumerate(self.joint_names):
                d[f"diagnostics/pos_target_bound/{jname}_upper"] = self.pos_target_bound_tracker.upper[i]
                d[f"diagnostics/pos_target_bound/{jname}_lower"] = self.pos_target_bound_tracker.lower[i]
        if self.track_vel_target_bounds:
            for i, jname in enumerate(self.joint_names):
                d[f"diagnostics/vel_target_bound/{jname}_upper"] = self.vel_target_bound_tracker.upper[i]
                d[f"diagnostics/vel_target_bound/{jname}_lower"] = self.vel_target_bound_tracker.lower[i]
        return d
    
    @property
    def action_dim(self):
        return len(self.joint_ids)

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
        self.action_buf = self.action_buf.roll(1, dims=1)
        self.action_buf[:, 0] = action
        delay_mask = (
            torch.arange(self.action_queue.shape[1], device=self.device)
            < self.delay
        ).reshape(self.num_envs, self.action_queue.shape[1], 1)
        self.action_queue = torch.where(delay_mask, self.action_queue, action.unsqueeze(1))

    @override
    def symmetry_transform(self):
        return joint_space_symmetry(self.asset, self.joint_names)


class JointPosition(_DelayedJointAction):
    """Absolute joint-position offset controller.

    This action maps policy outputs to a target posture each substep:
    `target = default_joint_pos + action * action_scaling` (on controlled joints),
    with optional random delay and first-order smoothing (LPF via `alpha`).

    Use this when you want the policy to command pose offsets directly around the
    nominal/default posture, without integrating action over time.
    """
    def __init__(
        self,
        env,
        action_scaling: Dict[str, float] = 0.5,
        max_delay: int = 2,
        alpha_range: Tuple[float, float] = (0.5, 1.0),
        track_pos_target_bounds: bool = False,
    ):
        super().__init__(
            env,
            action_scaling=action_scaling,
            max_delay=max_delay,
            alpha_range=alpha_range,
            track_pos_target_bounds=track_pos_target_bounds,
            track_vel_target_bounds=False
        )
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids]
        self.offset = torch.zeros_like(self.default_joint_pos)
    
    def __repr__(self) -> str:
        return f"JointPosition(joint_names={self.joint_names}, joint_ids={self.joint_ids.tolist()})"

    @override
    def reset(self, env_ids: torch.Tensor):
        super().reset(env_ids)
        default_joint_pos = self.asset.data.default_joint_pos[env_ids.unsqueeze(1), self.joint_ids]
        self.default_joint_pos[env_ids] = default_joint_pos + self.offset[env_ids]

    @override
    def apply_action(self, substep: int):
        self.applied_action.lerp_(self.action_queue[:, 0], self.alpha)
        self.action_queue = self.action_queue.roll(-1, dims=1)

        jpos_target = self.default_joint_pos + self.applied_action * self.action_scaling
        self.asset.set_joint_position_target(jpos_target, joint_ids=self.joint_ids)

        if self.track_pos_target_bounds:
            self.pos_target_bound_tracker.update(jpos_target)


class JointPositionDelta(_DelayedJointAction):
    """Incremental (integrated) joint-position controller.

    Compared to `JointPosition`, this action integrates per-substep deltas:
    `target[t+1] = target[t] + clamp(action * action_scaling * physics_dt)`.
    The command still goes through delay and LPF (`alpha`) to better match
    hardware-like command filtering.

    Use this when you want rate-like behavior and smoother, trajectory-style
    evolution of joint targets instead of direct pose-offset commands.
    """
    def __init__(
        self,
        env,
        action_scaling: Dict[str, float] = 0.5,
        clamp_range: Tuple[float, float] = (-0.5 * torch.pi, 0.5 * torch.pi),
        max_delay: int = 2,
        alpha_range: Tuple[float, float] = (0.5, 1.0),
        track_pos_target_bounds: bool = False
    ):
        super().__init__(
            env,
            action_scaling,
            max_delay,
            alpha_range,
            track_pos_target_bounds=track_pos_target_bounds,
            track_vel_target_bounds=False
        )
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids].clone()
        self.clamp_range = tuple(clamp_range)
        self.jpos_target = self.default_joint_pos.clone()
    
    @override
    def reset(self, env_ids: torch.Tensor):
        super().reset(env_ids)
        self.jpos_target[env_ids] = self.default_joint_pos[env_ids]
    
    @override
    def apply_action(self, substep: int):
        self.applied_action.lerp_(self.action_queue[:, 0], self.alpha)
        self.action_queue = self.action_queue.roll(-1, dims=1)

        delta = self.applied_action * self.action_scaling * self.env.physics_dt
        self.jpos_target += torch.clamp(delta, self.clamp_range[0], self.clamp_range[1])
        self.asset.set_joint_position_target(self.jpos_target, joint_ids=self.joint_ids)

        if self.track_pos_target_bounds:
            self.pos_target_bound_tracker.update(self.jpos_target)


class JointVelocity(_DelayedJointAction):

    def __init__(
        self,
        env,
        action_scaling: Dict[str, float] = 0.5,
        max_delay: int = 2,
        alpha_range: Tuple[float, float] = (0.5, 1.0),
        track_vel_target_bounds: bool = False
    ):
        super().__init__(
            env,
            action_scaling,
            max_delay,
            alpha_range,
            track_pos_target_bounds=False,
            track_vel_target_bounds=track_vel_target_bounds
        )
    
    @override
    def apply_action(self, substep: int):
        self.applied_action.lerp_(self.action_queue[:, 0], self.alpha)
        self.action_queue = self.action_queue.roll(-1, dims=1)

        jvel_target = self.applied_action * self.action_scaling
        self.asset.set_joint_velocity_target(jvel_target, joint_ids=self.joint_ids)
        
        if self.track_vel_target_bounds:
            self.vel_target_bound_tracker.update(jvel_target)


class CorrelatedJointPosition(Action):
    """Map a low-dimensional action to correlated joint position targets.

    Each controlled joint receives ``default_joint_pos + action_scaling * (action @ matrix.T)``,
    where ``matrix`` has shape ``(num_joints, action_dim)``. This is useful for parallel
    grippers where one scalar command should open/close multiple joints with opposite sign.
    """

    def __init__(
        self,
        env,
        joint_names: str | list[str],
        matrix: list[float] | list[list[float]],
        action_scaling: float = 1.0,
    ) -> None:
        super().__init__(env)
        joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.joint_ids = torch.tensor(joint_ids, device=self.device)

        coeffs = torch.tensor(matrix, dtype=torch.float32, device=self.device)
        if coeffs.ndim == 1:
            coeffs = coeffs.unsqueeze(-1)
        if coeffs.shape[0] != len(self.joint_names):
            raise ValueError(
                f"matrix rows ({coeffs.shape[0]}) must match number of joints "
                f"({len(self.joint_names)})"
            )
        self.matrix = coeffs
        self.action_dim = int(self.matrix.shape[1])
        self.action_scaling = action_scaling

        with torch.device(self.device):
            self.default_joint_pos = self.asset.data.default_joint_pos[
                :, self.joint_ids
            ].clone()
            self.action_buf = torch.zeros(self.num_envs, 4, self.action_dim)
            self.applied_action = torch.zeros(self.num_envs, self.action_dim)

    def __repr__(self) -> str:
        return (
            f"CorrelatedJointPosition(joint_names={self.joint_names}, "
            f"joint_ids={self.joint_ids.tolist()}, action_dim={self.action_dim})"
        )

    @override
    def reset(self, env_ids: torch.Tensor) -> None:
        self.action_buf[env_ids] = 0.0
        self.applied_action[env_ids] = 0.0
        self.default_joint_pos[env_ids] = self.asset.data.default_joint_pos[
            env_ids.unsqueeze(1), self.joint_ids
        ]

    @override
    def process_action(self, action: torch.Tensor) -> None:
        self.action_buf = self.action_buf.roll(1, dims=1)
        self.action_buf[:, 0] = action
        self.applied_action = action

    @override
    def apply_action(self, substep: int) -> None:
        joint_delta = (self.applied_action @ self.matrix.T) * self.action_scaling
        jpos_target = self.default_joint_pos + joint_delta
        self.asset.set_joint_position_target(jpos_target, joint_ids=self.joint_ids)

    @override
    def symmetry_transform(self):
        if not self.action_dim == 1:
            raise NotImplementedError("Symmetry transform is not supported for correlated joint positions with action dimension > 1")
        return SymmetryTransform(
            perm=torch.arange(self.action_dim),
            signs=[1] * self.action_dim,
        )


__all__ = [
    "JointPosition",
    "JointPositionDelta",
    "JointVelocity",
    "CorrelatedJointPosition",
    "SoftBoundTracker",
]
