from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from tensordict import TensorDictBase
from typing_extensions import override

from active_adaptation.planning import QuinticPolynomial
from active_adaptation.utils.math import quat_rotate_inverse, wrap_to_pi, yaw_quat

from active_adaptation.envs.mdp.commands.base import CommandV2
from active_adaptation.envs.mdp.rewards.base import RewardV2
from active_adaptation.envs.mdp.terminations.base import TerminationV2

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase

# Horizontal speed below which we keep the previous / fallback yaw.
_YAW_EPS = 1e-3


def _as_range3(
    ranges: Sequence[Sequence[float]],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    assert len(ranges) == 3, f"expected 3 axis ranges, got {len(ranges)}"
    return tuple((float(lo), float(hi)) for lo, hi in ranges)  # type: ignore[return-value]


def _yaw_from_xy(dx: torch.Tensor, dy: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    """Batched ``atan2(dy, dx)`` with fallback when the horizontal direction is tiny.

    ``dx`` / ``dy`` / returned yaw share shape ``(...)``; ``fallback`` is broadcast to that
    (accepts a trailing singleton, e.g. ``(N, 1)`` with ``dx`` of shape ``(N,)`` or ``(N, K)``).
    """
    yaw = torch.atan2(dy, dx)
    valid = (dx * dx + dy * dy).sqrt() > _YAW_EPS
    fb = fallback
    while fb.ndim > yaw.ndim:
        fb = fb.squeeze(-1)
    return torch.where(valid, yaw, fb.expand_as(yaw))


class TrajTracking(CommandV2):
    """Track a batched 3D root trajectory (quintic, zero endpoint acceleration).

    On reset, samples an endpoint ``(xT, vT)`` and duration and fits a
    :class:`~active_adaptation.planning.QuinticPolynomial` from the current
    root state. Next-step references are written only in :meth:`update`
    (not ``sync_state`` / ``reset``): the post-reset observation is discarded
    via ``is_init``, and rewards read the previous step's next-step targets.

    Orientation is yaw-only, derived from ``orientation_mode``:
    - ``0``: face the reference horizontal velocity
    - ``1``: face the trajectory endpoint (goal)
    """

    def __init__(
        self,
        target_range: Sequence[Sequence[float]],
        future_steps: list[int],
        vel_range: Sequence[Sequence[float]] = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        duration_range: tuple[float, float] = (2.0, 5.0),
        look_at_goal_prob: float = 0.5,
        resample_interval: int = 100,
        resample_prob: float = 0.2,
    ) -> None:
        super().__init__()
        self.target_range = _as_range3(target_range)
        self.vel_range = _as_range3(vel_range)
        self.duration_range = (float(duration_range[0]), float(duration_range[1]))
        self.future_steps = list(future_steps)
        self.look_at_goal_prob = float(look_at_goal_prob)
        self.resample_interval = int(resample_interval)
        self.resample_prob = float(resample_prob)

    @override
    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)

        with torch.device(self.device):
            self.future_steps_t = torch.tensor(self.future_steps, dtype=torch.long)
            self.ref_pos_w = torch.zeros(self.num_envs, 3)
            self.ref_lin_vel_w = torch.zeros(self.num_envs, 3)
            self.ref_yaw_w = torch.zeros(self.num_envs, 1)
            self.goal_pos_w = torch.zeros(self.num_envs, 3)
            self.target_pos_w = torch.zeros(self.num_envs, len(self.future_steps), 3)
            self.target_lin_vel_w = torch.zeros(self.num_envs, len(self.future_steps), 3)
            self.target_pos_b = torch.zeros(self.num_envs, len(self.future_steps), 3)
            self.target_lin_vel_b = torch.zeros(self.num_envs, len(self.future_steps), 3)
            self.target_yaw_w = torch.zeros(self.num_envs, len(self.future_steps), 1)
            self.target_yaw_err = torch.zeros(self.num_envs, len(self.future_steps), 1)
            self._cum_error = torch.zeros(self.num_envs, 1)
            self.Kp = torch.zeros(self.num_envs, 1)
            # mode 0: always look in the velocity direction
            # mode 1: look at the target position
            self.orientation_mode = torch.zeros(self.num_envs, 1, dtype=torch.long)

            self.traj = QuinticPolynomial.create(
                torch.zeros(self.num_envs, 3),
                torch.zeros(self.num_envs, 3),
                torch.zeros(self.num_envs, 3),
                torch.zeros(self.num_envs, 3),
                duration=1.0,
            )

    def _sample_vec(
        self,
        n: int,
        ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    ) -> torch.Tensor:
        out = torch.empty(n, 3, device=self.device)
        for i, (lo, hi) in enumerate(ranges):
            out[:, i].uniform_(lo, hi)
        return out

    def _yaw_command(
        self,
        pos_w: torch.Tensor,
        vel_w: torch.Tensor,
        fallback_yaw: torch.Tensor,
    ) -> torch.Tensor:
        """Resolve yaw command for shape ``(N, 3)`` or ``(N, K, 3)`` → ``(..., 1)``.

        Mode 0: horizontal velocity heading. Mode 1: look from ``pos_w`` toward ``goal_pos_w``.
        """
        if pos_w.ndim == 2:
            goal_delta = self.goal_pos_w - pos_w
            mode = self.orientation_mode.squeeze(-1)  # (N,)
        else:
            goal_delta = self.goal_pos_w.unsqueeze(1) - pos_w
            mode = self.orientation_mode  # (N, 1) broadcasts over K

        yaw_vel = _yaw_from_xy(vel_w[..., 0], vel_w[..., 1], fallback_yaw)
        yaw_goal = _yaw_from_xy(goal_delta[..., 0], goal_delta[..., 1], fallback_yaw)
        return torch.where(mode == 0, yaw_vel, yaw_goal).unsqueeze(-1)

    @property
    def command(self) -> torch.Tensor:
        """Body-yaw pos/vel at each horizon, plus yaw errors to the commanded heading."""
        return torch.cat(
            [
                self.target_pos_b.reshape(self.num_envs, -1),
                self.target_lin_vel_b.reshape(self.num_envs, -1),
                self.target_yaw_err.reshape(self.num_envs, -1),
            ],
            dim=-1,
        )

    @override
    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Spawn at default root pose; ``reset`` fits the trajectory from here."""
        return super().sample_init(env_ids)
    
    def _resample_target(self, env_ids: torch.Tensor) -> None:
        x0 = self.asset.data.root_link_pos_w[env_ids]
        v0 = self.asset.data.root_link_lin_vel_w[env_ids]
        xT = x0 + self._sample_vec(len(env_ids), self.target_range)
        vT = self._sample_vec(len(env_ids), self.vel_range)

        duration = torch.empty(len(env_ids), 1, device=self.device)
        duration.uniform_(*self.duration_range)

        self.traj[env_ids] = QuinticPolynomial.create(x0, v0, xT, vT, duration=duration)
        self.goal_pos_w[env_ids] = xT
        self.orientation_mode[env_ids] = (
            torch.rand(len(env_ids), 1, device=self.device) < self.look_at_goal_prob
        ).long()

    @override
    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase) -> None:
        self._resample_target(env_ids)
        Kp = torch.empty(len(env_ids), 1, device=self.device)
        Kp.uniform_(0.5, 1.0)
        self.Kp[env_ids] = Kp
        self._cum_error[env_ids] = 0.0
        # Do not evaluate targets here: the first post-reset obs is discarded
        # (``is_init``); next-step targets are written in ``update``.

    @override
    def sync_state(self) -> None:
        # Rewards read next-step targets from the previous ``update``.
        self._cum_error = (self.ref_pos_w - self.asset.data.root_link_pos_w).norm(
            dim=-1, keepdim=True
        )

    @override
    def update(self) -> None:
        resample = (
            ((self.env.episode_length_buf + 1) % self.resample_interval == 0) &
            (torch.rand(self.num_envs, device=self.device) < self.resample_prob)
        )
        if resample.any():
            resample_ids = resample.nonzero().squeeze(-1)
            self._resample_target(resample_ids) 

        # Next-step references for the upcoming observation / following reward.
        t_next = (self.env.episode_length_buf + 1).unsqueeze(1) * self.env.step_dt
        t_next = torch.minimum(t_next, self.traj.duration)
        ref_pos, ref_vel = self.traj.eval(t_next) # [N, 1, 3]
        
        self.ref_pos_w = ref_pos.squeeze(1) # [N, 3]
        pos_error = self.ref_pos_w - self.asset.data.root_link_pos_w
        self.ref_lin_vel_w = ref_vel.squeeze(1) + self.Kp * pos_error # [N, 3]

        heading = self.asset.data.heading_w  # (N,)
        self.ref_yaw_w = self._yaw_command(
            self.ref_pos_w,
            self.ref_lin_vel_w,
            heading,
        )

        t_query = (self.env.episode_length_buf.unsqueeze(1) + self.future_steps_t) * self.env.step_dt
        t_query = torch.minimum(t_query, self.traj.duration)
        pos_w, vel_w = self.traj.eval(t_query)
        self.target_pos_w = pos_w
        self.target_lin_vel_w = vel_w

        root_pos = self.asset.data.root_link_pos_w.unsqueeze(1)
        quat = yaw_quat(self.asset.data.root_link_quat_w).unsqueeze(1)
        self.target_pos_b = quat_rotate_inverse(quat, pos_w - root_pos)
        self.target_lin_vel_b = quat_rotate_inverse(quat, vel_w)

        self.target_yaw_w = self._yaw_command(
            pos_w,
            vel_w,
            heading.unsqueeze(1),
        )
        self.target_yaw_err = wrap_to_pi(self.target_yaw_w - heading.view(-1, 1, 1))

    @override
    def debug_draw(self) -> None:
        if not self.env.sim.has_gui():
            return

        T0 = self.traj.duration[0:1]
        ts = torch.linspace(0.0, 1.0, 32, device=self.device).unsqueeze(0) * T0
        xs, _ = self.traj[0:1].eval(ts)
        self.env.scene.draw_plot(xs[0], color=(1.0, 1.0, 1.0, 1.0), size=2.0)

        self.env.scene.draw_point(self.ref_pos_w, color=(1.0, 0.4, 0.1, 1.0), size=16.0)
        self.env.scene.draw_point(
            self.target_pos_w.reshape(-1, 3),
            color=(0.1, 0.85, 0.95, 1.0),
            size=12.0,
        )
        self.env.scene.draw_vector(
            self.ref_pos_w,
            self.ref_lin_vel_w,
            color=(0.2, 1.0, 0.3, 1.0),
            size=2.0,
        )
        # Commanded yaw as a horizontal unit arrow.
        yaw_dir = torch.stack(
            [self.ref_yaw_w.cos().squeeze(-1), self.ref_yaw_w.sin().squeeze(-1), torch.zeros(self.num_envs, device=self.device)],
            dim=-1,
        )
        self.env.scene.draw_vector(
            self.ref_pos_w,
            yaw_dir,
            color=(1.0, 0.85, 0.1, 1.0),
            size=2.0,
        )


class position_tracking(RewardV2[TrajTracking]):
    """Exponential reward for tracking the commanded next-step trajectory position."""

    def __init__(
        self,
        weight: float,
        enabled: bool = True,
        track_var: bool = False,
        sigma: float = 0.25,
    ):
        super().__init__(weight, enabled=enabled, track_var=track_var)
        self.sigma = sigma

    @override
    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)
        self.asset = self.env.scene.articulations["robot"]

    @override
    def _compute(self) -> torch.Tensor:
        error = (
            self.asset.data.root_link_pos_w - self.command_manager.ref_pos_w
        ).square().sum(dim=-1, keepdim=True)
        return torch.exp(-error / self.sigma)


class velocity_tracking(RewardV2[TrajTracking]):
    """Exponential reward for tracking the commanded next-step trajectory velocity."""

    def __init__(
        self,
        weight: float,
        enabled: bool = True,
        track_var: bool = False,
        sigma: float = 0.25,
    ):
        super().__init__(weight, enabled=enabled, track_var=track_var)
        self.sigma = sigma

    @override
    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)
        self.asset = self.env.scene.articulations["robot"]

    @override
    def _compute(self) -> torch.Tensor:
        error = (
            self.asset.data.root_link_lin_vel_w - self.command_manager.ref_lin_vel_w
        ).square().sum(dim=-1, keepdim=True)
        return torch.exp(-error / self.sigma)


class orientation_tracking(RewardV2[TrajTracking]):
    """Exponential reward for tracking the commanded next-step yaw."""

    def __init__(
        self,
        weight: float,
        enabled: bool = True,
        track_var: bool = False,
        sigma: float = 0.5,
    ):
        super().__init__(weight, enabled=enabled, track_var=track_var)
        self.sigma = sigma

    @override
    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)
        self.asset = self.env.scene.articulations["robot"]

    @override
    def _compute(self) -> torch.Tensor:
        yaw_err = wrap_to_pi(
            self.asset.data.heading_w.unsqueeze(1) - self.command_manager.ref_yaw_w
        )
        return torch.exp(-yaw_err.square() / self.sigma)


class position_error_exceeding(TerminationV2[TrajTracking]):
    """Terminate if the position error exceeds a threshold."""

    def __init__(
        self,
        threshold: float = 0.5,
    ):
        super().__init__(threshold)
        self.threshold = threshold

    @override
    def compute(self, termination: torch.Tensor):
        return termination | (self.command_manager._cum_error > self.threshold)
