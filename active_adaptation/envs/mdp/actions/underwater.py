from __future__ import annotations

import torch

from typing import TYPE_CHECKING, Sequence, Tuple, cast
from typing_extensions import override

from active_adaptation.utils.math import quat_rotate_inverse
from active_adaptation.utils.symmetry import SymmetryTransform

from .base import Action
from tensordict import TensorDictBase

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase
    from active_adaptation.envs.behaviors.underwater import UnderwaterRobot, UnderwaterRobotData


def _gain(value: float | Sequence[float], dim: int, device: torch.device) -> torch.Tensor:
    if isinstance(value, (int, float)):
        return torch.full((1, dim), float(value), device=device)
    gain = torch.as_tensor(value, device=device, dtype=torch.float32).reshape(1, -1)
    if gain.shape[-1] != dim:
        raise ValueError(f"Expected {dim} gains, got shape {tuple(gain.shape)}")
    return gain


def _root_twist_b(asset, backend: str) -> torch.Tensor:
    data = asset.data
    quat = data.root_link_quat_w
    if backend == "isaaclab":
        lin_w = data.root_com_lin_vel_w
        ang_w = data.root_com_ang_vel_w
    else:
        lin_w = data.root_link_lin_vel_w
        ang_w = data.root_link_ang_vel_w
    return torch.cat(
        [quat_rotate_inverse(quat, lin_w), quat_rotate_inverse(quat, ang_w)],
        dim=-1,
    )


class UnderwaterThrottle(Action):
    """Throttle action for underwater robots.

    The action directly controls per-rotor normalized throttle in ``[-1, 1]``.
    Throttle-to-thrust conversion is handled by ``UnderwaterRobot.write_data_to_sim``.
    """
    uw: "UnderwaterRobotData"

    def __init__(
        self,
        action_scaling: float = 1.0,
        alpha_range: Tuple[float, float] = (0.5, 1.0),
    ):
        super().__init__()
        self.action_scaling = float(action_scaling)
        self.alpha_range = tuple(alpha_range)

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        if not hasattr(self.asset, "data_underwater"):
            raise RuntimeError(
                "UnderwaterThrottle requires robot.data_underwater to be initialized."
            )
        self.uw = cast("UnderwaterRobotData", self.asset.data_underwater)
        self.action_dim = int(self.uw.throttle_cmd.shape[-1])
        self.action_buf = torch.zeros(self.num_envs, 4, self.action_dim, device=self.device)
        self.applied_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.alpha = torch.ones(self.num_envs, 1, device=self.device)
        self.names = [f"rotor_{i}" for i in range(self.action_dim)]

    @override
    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase):
        alpha = torch.empty(len(env_ids), 1, device=self.device)
        alpha.uniform_(self.alpha_range[0], self.alpha_range[1])
        self.alpha[env_ids] = alpha
        self.action_buf[env_ids] = 0.0
        self.applied_action[env_ids] = 0.0
        self.uw.throttle_cmd[env_ids] = 0.0
        self.uw.throttle[env_ids] = 0.0

    @override
    def process_action(self, action: torch.Tensor | None):
        if action is None:
            return
        self.action_buf = self.action_buf.roll(1, dims=1)
        self.action_buf[:, 0] = action

    @override
    def apply_action(self, substep: int):
        self.applied_action.lerp_(self.action_buf[:, 0], self.alpha)
        self.uw.throttle_cmd.copy_(
            torch.clamp(self.applied_action * self.action_scaling, -1.0, 1.0)
        )

    @override
    def symmetry_transform(self):
        return SymmetryTransform(
            perm=torch.arange(self.action_dim),
            signs=[1] * self.action_dim,
        )


class _AllocatedTwistAction(Action):
    """Non-RL baseline: body twist → PD wrench → optional allocation → throttle."""

    uw: "UnderwaterRobotData"
    underwater: "UnderwaterRobot"

    def __init__(
        self,
        action_dim: int,
        names: Sequence[str],
        kp: float | Sequence[float] = 20.0,
        kd: float | Sequence[float] = 2.0,
        alpha_range: Tuple[float, float] = (0.5, 1.0),
    ):
        super().__init__()
        self._action_dim = int(action_dim)
        self._name_list = list(names)
        self._kp = kp
        self._kd = kd
        self.alpha_range = tuple(alpha_range)

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.underwater = env.require_behavior("underwater")
        self.uw = cast("UnderwaterRobotData", self.asset.data_underwater)
        self.action_dim = self._action_dim
        self.action_buf = torch.zeros(self.num_envs, 4, self.action_dim, device=self.device)
        self.applied_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.alpha = torch.ones(self.num_envs, 1, device=self.device)
        self.kp = _gain(self._kp, 6, self.device)
        self.kd = _gain(self._kd, 6, self.device)
        self.names = self._name_list

    @override
    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase):
        alpha = torch.empty(len(env_ids), 1, device=self.device)
        alpha.uniform_(self.alpha_range[0], self.alpha_range[1])
        self.alpha[env_ids] = alpha
        self.action_buf[env_ids] = 0.0
        self.applied_action[env_ids] = 0.0
        self.uw.throttle_cmd[env_ids] = 0.0
        self.uw.throttle[env_ids] = 0.0

    @override
    def process_action(self, action: torch.Tensor | None):
        if action is None:
            return
        self.action_buf = self.action_buf.roll(1, dims=1)
        self.action_buf[:, 0] = action

    def _apply_twist_cmd(self, twist_cmd: torch.Tensor) -> None:
        twist_b = _root_twist_b(self.asset, self.env.backend)
        err = twist_cmd - twist_b
        wrench_b = self.kp * err - self.kd * twist_b
        self.uw.throttle_cmd.copy_(self.underwater.allocate_wrench(wrench_b))

    @override
    def symmetry_transform(self):
        return SymmetryTransform(
            perm=torch.arange(self.action_dim),
            signs=[1] * self.action_dim,
        )


class UnderwaterVelocity(_AllocatedTwistAction):
    """Body-frame twist tracker for non-RL baselines.

    Action is ``[v_b, ω_b]`` with shape ``(N, 6)``. Each substep runs a PD law
    on twist error, then the optional allocation matrix maps wrench to
    ``throttle_cmd``. RL policies should use :class:`UnderwaterThrottle`.
    """

    def __init__(
        self,
        kp: float | Sequence[float] = 20.0,
        kd: float | Sequence[float] = 2.0,
        alpha_range: Tuple[float, float] = (0.5, 1.0),
    ):
        super().__init__(
            action_dim=6,
            names=["vx", "vy", "vz", "wx", "wy", "wz"],
            kp=kp,
            kd=kd,
            alpha_range=alpha_range,
        )

    @override
    def apply_action(self, substep: int):
        self.applied_action.lerp_(self.action_buf[:, 0], self.alpha)
        self._apply_twist_cmd(self.applied_action)
    
    @override
    def debug_draw(self):
        pass


class UnderwaterPositionVelocity(_AllocatedTwistAction):
    """Body-frame position + twist tracker for non-RL baselines.

    Action is ``[p_b, v_b, ω_b]`` with shape ``(N, 9)``. ``p_b`` is treated as
    a body-frame position error (or commanded offset). Outer P:
    ``v_ref = kp_pos * p_b + v_b``, then the same twist PD / allocation as
    :class:`UnderwaterVelocity`.
    """

    def __init__(
        self,
        kp: float | Sequence[float] = 20.0,
        kd: float | Sequence[float] = 2.0,
        kp_pos: float | Sequence[float] = 1.0,
        alpha_range: Tuple[float, float] = (0.5, 1.0),
    ):
        super().__init__(
            action_dim=9,
            names=["px", "py", "pz", "vx", "vy", "vz", "wx", "wy", "wz"],
            kp=kp,
            kd=kd,
            alpha_range=alpha_range,
        )
        self._kp_pos = kp_pos

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.kp_pos = _gain(self._kp_pos, 3, self.device)

    @override
    def apply_action(self, substep: int):
        self.applied_action.lerp_(self.action_buf[:, 0], self.alpha)
        pos_b = self.applied_action[:, :3]
        vel_b = self.applied_action[:, 3:6]
        ang_b = self.applied_action[:, 6:9]
        v_ref = self.kp_pos * pos_b + vel_b
        self._apply_twist_cmd(torch.cat([v_ref, ang_b], dim=-1))
    
    @override
    def debug_draw(self):
        pass


__all__ = ["UnderwaterThrottle", "UnderwaterVelocity", "UnderwaterPositionVelocity"]
