from __future__ import annotations

import math
import torch
import torch.nn as nn

from typing import TYPE_CHECKING, Dict, Optional, Tuple
from typing_extensions import override

from active_adaptation.utils.string import resolve_matching_names_values
from active_adaptation.utils.symmetry import SymmetryTransform, joint_space_symmetry

from .base import ActionV2
from tensordict import TensorDictBase


if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


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


class _DelayedJointAction(ActionV2):
    def __init__(
        self,
        action_scaling: Dict[str, float] | float = 0.5,
        max_delay: int = 2,
        alpha_range: Tuple[float, float] = (0.5, 1.0),
        track_pos_target_bounds: bool = False,
        track_vel_target_bounds: bool = False,
    ):
        super().__init__()
        self.track_pos_target_bounds = track_pos_target_bounds
        self.track_vel_target_bounds = track_vel_target_bounds

        if isinstance(action_scaling, float):
            action_scaling = {".*": float(action_scaling)}
        self._action_scaling = dict(action_scaling)
        self.max_delay = max_delay
        self.alpha_range = tuple(alpha_range)

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)

        _, self.joint_names, scaling = resolve_matching_names_values(
            self._action_scaling, self.asset.cfg.joint_names_simulation
        )
        self.joint_ids = torch.tensor(
            [self.asset.joint_names.index(name) for name in self.joint_names],
            device=self.device,
        )
        self.names = self.joint_names

        self.action_scaling = torch.tensor(scaling, device=self.device)
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
    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase):
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
        action_scaling: Dict[str, float] | float = 0.5,
        max_delay: int = 2,
        alpha_range: Tuple[float, float] = (0.5, 1.0),
        track_pos_target_bounds: bool = False,
    ):
        super().__init__(
            action_scaling=action_scaling,
            max_delay=max_delay,
            alpha_range=alpha_range,
            track_pos_target_bounds=track_pos_target_bounds,
            track_vel_target_bounds=False,
        )

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids]
        self.offset = torch.zeros_like(self.default_joint_pos)

    def __repr__(self) -> str:
        return f"JointPosition(joint_names={self.joint_names}, joint_ids={self.joint_ids.tolist()})"

    @override
    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase):
        super().reset(env_ids, tensordict)
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


class JointReferenceModel(_DelayedJointAction):
    """Second-order reference-model prefilter for PD joint targets.

    Raw policy actions drive a fixed second-order low-pass filter whose state
    ``(q_bar, v_bar)`` is sent to the PD controller as position/velocity
    targets::

        target = q0 + action_scaling * a
        v_dot = omega^2 * (target - q_bar) - 2*zeta*omega * v_bar
        v_bar <- v_bar + dt * v_dot
        q_bar <- q_bar + dt * v_bar

    ``omega`` (natural frequency / bandwidth, rad/s) and ``zeta`` (damping
    ratio) are fixed constants, not policy outputs. Critical damping is
    ``zeta = 1`` (``omega`` only sets how fast the filter responds, not
    whether it is overdamped or underdamped). Use ``zeta >= 1`` to avoid
    resonant peaking in the reference trajectory. Filter state resets to the
    current joint positions with ``v_bar = 0`` on episode start to avoid
    startle transients.
    """

    def __init__(
        self,
        action_scaling: Dict[str, float] | float = 0.5,
        omega: Dict[str, float] | float = 20.0,
        zeta: Dict[str, float] | float = 1.0 / math.sqrt(2),
        max_delay: int = 2,
        alpha_range: Tuple[float, float] = (1.0, 1.0),
        track_pos_target_bounds: bool = False,
        track_vel_target_bounds: bool = False,
    ):
        super().__init__(
            action_scaling=action_scaling,
            max_delay=max_delay,
            alpha_range=alpha_range,
            track_pos_target_bounds=track_pos_target_bounds,
            track_vel_target_bounds=track_vel_target_bounds,
        )
        if isinstance(omega, (int, float)):
            omega = {".*": float(omega)}
        if isinstance(zeta, (int, float)):
            zeta = {".*": float(zeta)}
        self._omega = dict(omega)
        self._zeta = dict(zeta)

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids]
        self.offset = torch.zeros_like(self.default_joint_pos)

        _, _, omega = resolve_matching_names_values(
            self._omega, self.joint_names
        )
        _, _, zeta = resolve_matching_names_values(
            self._zeta, self.joint_names
        )
        self.omega = torch.tensor(omega, device=self.device)
        self.zeta = torch.tensor(zeta, device=self.device)

        self.q_bar = self.default_joint_pos.clone()
        self.v_bar = torch.zeros_like(self.default_joint_pos)

    def __repr__(self) -> str:
        return (
            f"JointReferenceModel(joint_names={self.joint_names}, "
            f"joint_ids={self.joint_ids.tolist()})"
        )

    @override
    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase):
        super().reset(env_ids, tensordict)
        default_joint_pos = self.asset.data.default_joint_pos[
            env_ids.unsqueeze(1), self.joint_ids
        ]
        self.default_joint_pos[env_ids] = default_joint_pos + self.offset[env_ids]
        self.q_bar[env_ids] = self.default_joint_pos[env_ids]
        self.v_bar[env_ids] = 0.0

    @override
    def apply_action(self, substep: int):
        self.applied_action.lerp_(self.action_queue[:, 0], self.alpha)
        self.action_queue = self.action_queue.roll(-1, dims=1)

        target = self.default_joint_pos + self.applied_action * self.action_scaling
        v_dot = (
            self.omega.square() * (target - self.q_bar)
            - 2.0 * self.zeta * self.omega * self.v_bar
        )
        dt = self.env.physics_dt
        self.v_bar.add_(dt * v_dot)
        self.q_bar.add_(dt * self.v_bar)

        self.asset.set_joint_position_target(self.q_bar, joint_ids=self.joint_ids)
        self.asset.set_joint_velocity_target(self.v_bar, joint_ids=self.joint_ids)

        if self.track_pos_target_bounds:
            self.pos_target_bound_tracker.update(self.q_bar)
        if self.track_vel_target_bounds:
            self.vel_target_bound_tracker.update(self.v_bar)


class JointLeakyVelocityModel(_DelayedJointAction):
    """Two-stage leaky velocity reference model for PD joint targets.

    Raw policy actions drive a first-order velocity tracker with a leaky
    position integrator::

        v_dot = omega * (action_scaling * a - v_bar)
        v_bar <- v_bar + dt * v_dot
        q_bar <- q_bar + dt * (v_bar - leak_rate * (q_bar - q0))

    ``action_scaling``, ``omega``, and ``leak_rate`` are fixed constants, not
    policy outputs. Steady-state position reach from action scales as
    ``action_scaling / leak_rate``, independent of ``omega``. Filter state
    resets to current joint positions with ``v_bar = 0`` on episode start.
    """

    def __init__(
        self,
        action_scaling: Dict[str, float] | float = 0.5,
        omega: Dict[str, float] | float = 20.0,
        leak_rate: Dict[str, float] | float = 2.0,
        max_delay: int = 2,
        alpha_range: Tuple[float, float] = (1.0, 1.0),
        track_pos_target_bounds: bool = False,
        track_vel_target_bounds: bool = False,
    ):
        super().__init__(
            action_scaling=action_scaling,
            max_delay=max_delay,
            alpha_range=alpha_range,
            track_pos_target_bounds=track_pos_target_bounds,
            track_vel_target_bounds=track_vel_target_bounds,
        )
        self._omega = self._as_joint_dict(omega)
        self._leak_rate = self._as_joint_dict(leak_rate)

    @staticmethod
    def _as_joint_dict(value: Dict[str, float] | float) -> Dict[str, float]:
        if isinstance(value, (int, float)):
            return {".*": float(value)}
        return dict(value)

    def _resolve_filter_params(self) -> Tuple[list[float], list[float], list[float]]:
        _, _, action_scaling = resolve_matching_names_values(
            self._action_scaling, self.joint_names
        )
        _, _, omega = resolve_matching_names_values(
            self._omega, self.joint_names
        )
        _, _, leak_rate = resolve_matching_names_values(
            self._leak_rate, self.joint_names
        )
        return action_scaling, omega, leak_rate

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids]
        self.offset = torch.zeros_like(self.default_joint_pos)

        action_scaling, omega, leak_rate = self._resolve_filter_params()
        self.action_scaling = torch.tensor(action_scaling, device=self.device)
        self.omega = torch.tensor(omega, device=self.device)
        self.leak_rate = torch.tensor(leak_rate, device=self.device)

        self.q_bar = self.default_joint_pos.clone()
        self.v_bar = torch.zeros_like(self.default_joint_pos)

    def __repr__(self) -> str:
        return (
            f"JointLeakyVelocityModel(joint_names={self.joint_names}, "
            f"joint_ids={self.joint_ids.tolist()})"
        )

    @override
    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase):
        super().reset(env_ids, tensordict)
        default_joint_pos = self.asset.data.default_joint_pos[
            env_ids.unsqueeze(1), self.joint_ids
        ]
        self.default_joint_pos[env_ids] = default_joint_pos + self.offset[env_ids]
        self.q_bar[env_ids] = self.asset.data.joint_pos[
            env_ids.unsqueeze(1), self.joint_ids
        ]
        self.v_bar[env_ids] = 0.0

    @override
    def apply_action(self, substep: int):
        self.applied_action.lerp_(self.action_queue[:, 0], self.alpha)
        self.action_queue = self.action_queue.roll(-1, dims=1)

        target_vel = self.applied_action * self.action_scaling
        v_dot = self.omega * (target_vel - self.v_bar)
        dt = self.env.physics_dt
        self.v_bar.add_(dt * v_dot)
        q_dot = self.v_bar - self.leak_rate * (self.q_bar - self.default_joint_pos)
        self.q_bar.add_(dt * q_dot)

        self.asset.set_joint_position_target(self.q_bar, joint_ids=self.joint_ids)
        self.asset.set_joint_velocity_target(self.v_bar, joint_ids=self.joint_ids)

        if self.track_pos_target_bounds:
            self.pos_target_bound_tracker.update(self.q_bar)
        if self.track_vel_target_bounds:
            self.vel_target_bound_tracker.update(self.v_bar)


class JointLeakyVelocityReachModel(JointLeakyVelocityModel):
    """Range-normalized leaky velocity model with a single ``t_reach`` knob.

    Same dynamics as :class:`JointLeakyVelocityModel`, but ``action_scaling``,
    ``omega``, and ``leak_rate`` are derived per joint from the joint range and
    a reach time ``t_reach`` (seconds to move near the joint limit under
    sustained unit action)::

        R = (q_high - q_low) / 2
        leak_rate = 1 / t_reach
        omega = min(10 * leak_rate, omega_pd / 5)   if omega_pd is given
                10 * leak_rate                      otherwise
        action_scaling = kappa * R * leak_rate

    ``kappa`` (default 0.8) is a safety margin so a sustained unit action
    reaches ``kappa * R`` rather than the physical limit.
    """

    def __init__(
        self,
        t_reach: Dict[str, float] | float = 0.5,
        kappa: float = 0.8,
        omega_pd: Dict[str, float] | float | None = None,
        max_delay: int = 2,
        alpha_range: Tuple[float, float] = (1.0, 1.0),
        track_pos_target_bounds: bool = False,
        track_vel_target_bounds: bool = False,
    ):
        t_reach_dict = self._as_joint_dict(t_reach)
        self._t_reach = t_reach_dict
        self._kappa = float(kappa)
        if omega_pd is not None:
            self._omega_pd = self._as_joint_dict(omega_pd)
        else:
            self._omega_pd = None

        super().__init__(
            action_scaling={key: 1.0 for key in t_reach_dict},
            omega={".*": 1.0},
            leak_rate={".*": 1.0},
            max_delay=max_delay,
            alpha_range=alpha_range,
            track_pos_target_bounds=track_pos_target_bounds,
            track_vel_target_bounds=track_vel_target_bounds,
        )

    def __repr__(self) -> str:
        return (
            f"JointLeakyVelocityReachModel(joint_names={self.joint_names}, "
            f"joint_ids={self.joint_ids.tolist()})"
        )

    @override
    def _resolve_filter_params(self) -> Tuple[list[float], list[float], list[float]]:
        _, _, t_reach = resolve_matching_names_values(
            self._t_reach, self.joint_names
        )
        limits = self.asset.data.soft_joint_pos_limits
        if limits is None:
            limits = self.asset.data.joint_pos_limits
        limits = limits[0, self.joint_ids]
        q_low, q_high = limits.unbind(-1)
        half_range = (q_high - q_low) / 2

        action_scaling: list[float] = []
        omega: list[float] = []
        leak_rate: list[float] = []
        omega_pd_vals: list[float] | None = None
        if self._omega_pd is not None:
            _, _, omega_pd_vals = resolve_matching_names_values(
                self._omega_pd, self.joint_names
            )

        for i, t in enumerate(t_reach):
            lam = 1.0 / t
            leak_rate.append(lam)
            if omega_pd_vals is not None:
                omega.append(min(10.0 * lam, omega_pd_vals[i] / 5.0))
            else:
                omega.append(10.0 * lam)
            action_scaling.append(self._kappa * float(half_range[i]) * lam)

        return action_scaling, omega, leak_rate


class JointPositionWithVelocityForward(_DelayedJointAction):
    """
    ExtremControl: Low-Latency Humanoid Teleoperation with Direct Extremity Control https://arxiv.org/pdf/2602.11321
    """

    def __init__(
        self,
        action_scaling: Dict[str, float] | float = 0.5,
        velocity_ff: float = 0.5,
        max_delay: int = 2,
        alpha_range: Tuple[float, float] = (0.5, 1.0),
        track_pos_target_bounds: bool = False,
    ):
        super().__init__(
            action_scaling=action_scaling,
            max_delay=max_delay,
            alpha_range=alpha_range,
            track_pos_target_bounds=track_pos_target_bounds,
            track_vel_target_bounds=False,
        )
        self.velocity_ff = velocity_ff

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids]
        self.offset = torch.zeros_like(self.default_joint_pos)
        # previous joint position target
        self._jpos_target = self.default_joint_pos.clone()

    @override
    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase):
        super().reset(env_ids, tensordict)
        default_joint_pos = self.asset.data.default_joint_pos[env_ids.unsqueeze(1), self.joint_ids]
        self.default_joint_pos[env_ids] = default_joint_pos + self.offset[env_ids]
        self._jpos_target[env_ids] = self.default_joint_pos[env_ids]

    @override
    def apply_action(self, substep: int):
        self.applied_action.lerp_(self.action_queue[:, 0], self.alpha)
        self.action_queue = self.action_queue.roll(-1, dims=1)

        jpos_target = self.default_joint_pos + self.applied_action * self.action_scaling
        jvel_target = self.velocity_ff * (jpos_target - self._jpos_target) / self.env.physics_dt
        self._jpos_target = jpos_target

        self.asset.set_joint_position_target(jpos_target, joint_ids=self.joint_ids)
        self.asset.set_joint_velocity_target(jvel_target, joint_ids=self.joint_ids)

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
        action_scaling: Dict[str, float] | float = 0.5,
        clamp_range: Tuple[float, float] = (-0.5 * torch.pi, 0.5 * torch.pi),
        max_delay: int = 2,
        alpha_range: Tuple[float, float] = (0.5, 1.0),
        track_pos_target_bounds: bool = False,
    ):
        super().__init__(
            action_scaling,
            max_delay,
            alpha_range,
            track_pos_target_bounds=track_pos_target_bounds,
            track_vel_target_bounds=False,
        )
        self.clamp_range = tuple(clamp_range)

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids].clone()
        self.jpos_target = self.default_joint_pos.clone()

    @override
    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase):
        super().reset(env_ids, tensordict)
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
        action_scaling: Dict[str, float] | float = 0.5,
        max_delay: int = 2,
        alpha_range: Tuple[float, float] = (0.5, 1.0),
        track_vel_target_bounds: bool = False,
    ):
        super().__init__(
            action_scaling,
            max_delay,
            alpha_range,
            track_pos_target_bounds=False,
            track_vel_target_bounds=track_vel_target_bounds,
        )

    @override
    def apply_action(self, substep: int):
        self.applied_action.lerp_(self.action_queue[:, 0], self.alpha)
        self.action_queue = self.action_queue.roll(-1, dims=1)

        jvel_target = self.applied_action * self.action_scaling
        self.asset.set_joint_velocity_target(jvel_target, joint_ids=self.joint_ids)

        if self.track_vel_target_bounds:
            self.vel_target_bound_tracker.update(jvel_target)


class CorrelatedJointPosition(ActionV2):
    """Map a low-dimensional action to correlated joint position targets.

    Each controlled joint receives ``default_joint_pos + action_scaling * (action @ matrix.T)``,
    where ``matrix`` has shape ``(num_joints, action_dim)``. This is useful for parallel
    grippers where one scalar command should open/close multiple joints with opposite sign.
    """

    def __init__(
        self,
        joint_names: str | list[str],
        matrix: list[float] | list[list[float]],
        action_scaling: float = 1.0,
    ) -> None:
        super().__init__()
        self.joint_names_expr = joint_names
        self._matrix = matrix
        self.action_scaling = action_scaling

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        joint_ids, self.joint_names = self.asset.find_joints(self.joint_names_expr)
        self.joint_ids = torch.tensor(joint_ids, device=self.device)

        coeffs = torch.tensor(self._matrix, dtype=torch.float32, device=self.device)
        if coeffs.ndim == 1:
            coeffs = coeffs.unsqueeze(-1)
        if coeffs.shape[0] != len(self.joint_names):
            raise ValueError(
                f"matrix rows ({coeffs.shape[0]}) must match number of joints "
                f"({len(self.joint_names)})"
            )
        self.matrix = coeffs
        self.action_dim = int(self.matrix.shape[1])
        self.names = [f"{self.joint_names_expr}_{i}" for i in range(self.action_dim)]

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
    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase) -> None:
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
    "JointReferenceModel",
    "JointLeakyVelocityModel",
    "JointLeakyVelocityReachModel",
    "JointPositionWithVelocityForward",
    "JointPositionDelta",
    "JointVelocity",
    "CorrelatedJointPosition",
    "SoftBoundTracker",
]
