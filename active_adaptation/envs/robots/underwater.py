"""
Backend-agnostic underwater robot wrapper around IsaacLab's Articulation or Mjlab's Entity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, TYPE_CHECKING
from functools import cached_property

import torch
from tensordict import TensorDictBase

from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse
from active_adaptation.utils.profiling import ScopedTimer
import active_adaptation.utils.string as string_utils

if TYPE_CHECKING: # DO NOT MODIFY
    # for the editor to work
    from isaaclab.assets import Articulation
    from active_adaptation.envs.env_base import _EnvBase


# Scalar volume, or damping as float / 3-tuple / 6-tuple per body.
BodyFloatSpec = Sequence[float] | Mapping[str, float]
BodyDampingValue = float | Sequence[float]
BodyDampingSpec = Sequence[BodyDampingValue] | Mapping[str, BodyDampingValue]


def _compute_root_flow(
    root_lin_vel_w: torch.Tensor,
    root_ang_vel_w: torch.Tensor,
    root_link_quat_w: torch.Tensor,
    flow_vels: torch.Tensor,
    flow_noise_scale: torch.Tensor,
    prev_body_vels: torch.Tensor,
    prev_body_acc: torch.Tensor,
    hydro_axis_sign: torch.Tensor,
    dt: float,
    acc_filter_alpha: float,
    add_noise: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute flow-relative root twist and filtered acceleration."""
    flow_twist_w = flow_vels
    if add_noise:
        flow_twist_w = flow_twist_w + torch.rand_like(flow_twist_w) * flow_noise_scale
    relative_twist_w = torch.cat(
        [
            root_lin_vel_w - flow_twist_w[..., :3],
            root_ang_vel_w - flow_twist_w[..., 3:],
        ],
        dim=-1,
    ).reshape(-1, 2, 3)
    root_quat_w = root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    hydro_twist_b = quat_rotate_inverse(root_quat_w, relative_twist_w).reshape_as(
        flow_twist_w
    )
    hydro_twist_b = hydro_twist_b * hydro_axis_sign
    hydro_acc_b = (hydro_twist_b - prev_body_vels) / dt
    hydro_acc_b = (
        (1.0 - acc_filter_alpha) * prev_body_acc
        + acc_filter_alpha * hydro_acc_b
    )
    return hydro_twist_b, hydro_acc_b, flow_twist_w


def _compute_buoyancy(
    body_quat_w: torch.Tensor,
    buoyancy_force: torch.Tensor,
    coBM: torch.Tensor,
    base_body_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute body-local buoyancy directly from WXYZ quaternions."""
    w, x, y, z = body_quat_w.unbind(-1)
    buoyancy_forces_b = torch.stack(
        [
            2.0 * buoyancy_force * (x * z - w * y),
            2.0 * buoyancy_force * (y * z + w * x),
            buoyancy_force * (1.0 - 2.0 * (x.square() + y.square())),
        ],
        dim=-1,
    )
    buoyancy_torques_b = torch.zeros_like(buoyancy_forces_b)
    buoyancy_torques_b[:, base_body_id, 0] = (
        -coBM * buoyancy_forces_b[:, base_body_id, 1]
    )
    buoyancy_torques_b[:, base_body_id, 1] = (
        coBM * buoyancy_forces_b[:, base_body_id, 0]
    )
    return buoyancy_forces_b, buoyancy_torques_b


def _expand_damping_coeffs(value: BodyDampingValue) -> list[float]:
    """Expand a damping value to a length-6 diagonal coefficient vector.

    - ``float`` → isotropic translational ``(d, d, d, 0, 0, 0)``
    - length-3 → translational anisotropic ``(fx, fy, fz, 0, 0, 0)``
    - length-6 → full wrench ``(Fx, Fy, Fz, Mx, My, Mz)``
    """
    if isinstance(value, (int, float)):
        d = float(value)
        return [d, d, d, 0.0, 0.0, 0.0]
    coeffs = [float(x) for x in value]
    if len(coeffs) == 3:
        return [coeffs[0], coeffs[1], coeffs[2], 0.0, 0.0, 0.0]
    if len(coeffs) == 6:
        return coeffs
    raise ValueError(
        f"Damping value must be a float or a sequence of length 3 or 6; got {value!r}"
    )


# Bodies introduced by mjlab fixed-base mocap wrapping; no hydrodynamics.
_HYDRO_SKIP_BODIES = frozenset({"mocap_base"})


def _hydro_required_bodies(body_names: Sequence[str]) -> list[str]:
    return [name for name in body_names if name not in _HYDRO_SKIP_BODIES]


@dataclass
class HydrodynamicsCfg:
    """Hydrodynamic parameters for an underwater robot.

    ``volume`` is per-body displaced volume (m^3). ``linear_damping`` /
    ``quadratic_damping`` are per-body diagonal coefficients. For each body pass:

    - a ``float`` (isotropic translational drag),
    - a length-3 sequence (anisotropic translational), or
    - a length-6 sequence (full Fossen-style wrench damping).

    Specs may be a sequence in ``robot.body_names`` order or a
    ``{name_regex: value}`` mapping. Every body must be specified.

    ``added_mass`` remains the vehicle-level 6-DoF term on the **base** twist.
    """
    volume: BodyFloatSpec
    coBM: float
    added_mass: tuple[float, float, float, float, float, float]
    linear_damping: BodyDampingSpec
    quadratic_damping: BodyDampingSpec
    water_density: float = 997.0
    gravity: float = 9.8
    acc_filter_alpha: float = 0.3


@dataclass
class UnderwaterRobotData:
    """Persistent and per-step underwater dynamics/propulsion buffers.

    Naming convention:
    - `*_b`: vector expressed in robot base/body frame.
    - 6D wrench vectors follow `[Fx, Fy, Fz, Mx, My, Mz]`.
    - Shapes are batched over environments (`num_envs`, ...).
    """
    # Constant (or slowly changing) hydrodynamics parameters/matrices.
    added_mass_matrix: torch.Tensor
    linear_damping: torch.Tensor  # (num_envs, num_bodies, 6)
    quadratic_damping: torch.Tensor  # (num_envs, num_bodies, 6)
    volume: torch.Tensor  # (num_envs, num_bodies)
    coBM: torch.Tensor

    # Temporal state for filtered body acceleration estimate.
    prev_body_vels: torch.Tensor
    prev_body_acc: torch.Tensor

    # Flow/current disturbance configuration and sampled flow state.
    flow_vels: torch.Tensor
    max_flow_vel: torch.Tensor
    flow_noise_scale: torch.Tensor

    # Rotor command/state used for throttle-to-thrust conversion.
    # `throttle_cmd` is action input (normalized [-1, 1]);
    # `throttle` is filtered actuator state.
    throttle_cmd: torch.Tensor
    throttle: torch.Tensor
    time_constants: torch.Tensor
    force_constants: torch.Tensor
    rpm: torch.Tensor
    thrusts_b: torch.Tensor

    # Per-step decomposed hydrodynamics terms.
    body_acc: torch.Tensor
    damping: torch.Tensor  # (num_envs, num_bodies, 6) body-frame wrenches
    added_mass: torch.Tensor
    coriolis: torch.Tensor
    buoyancy: torch.Tensor
    hydro: torch.Tensor

    # Final hydro wrench contribution applied to base link in body frame.
    hydro_forces_b: torch.Tensor
    hydro_torques_b: torch.Tensor


class UnderwaterRobot:
    def __init__(
        self,
        cfg: HydrodynamicsCfg,
        rotor_time_constants: Dict[str, float],
        rotor_force_constants: Dict[str, float],
        robot: "Articulation | None" = None,
        env: "_EnvBase | None" = None,
    ):
        self.cfg = cfg
        self._rotor_time_constants = dict(rotor_time_constants)
        self._rotor_force_constants = dict(rotor_force_constants)
        self.robot = None
        self.env = None
        self.dt = None
        self.rotor_names = []
        self.body_names = []
        self.rotor_indices = None
        self.data = None
        self._hydro_axis_sign = None
        self._flow_noise_enabled = False
        self._compute_buoyancy = _compute_buoyancy

        # Base body is assumed to be the root body in IsaacLab articulations.
        self._base_body_id = 0
        if robot is not None and env is not None:
            self._initialize(robot=robot, env=env)
        elif robot is not None or env is not None:
            raise ValueError("Both 'robot' and 'env' must be provided together.")

    @cached_property
    def num_envs(self) -> int:
        return self.env.num_envs

    @cached_property
    def device(self) -> torch.device:
        try:
            device = self.robot.device
        except AttributeError:
            device = self.robot.data.root_link_pos_w.device
        return device

    @cached_property
    def num_bodies(self) -> int:
        return len(self.body_names)

    def _resolve_body_floats(
        self,
        spec: BodyFloatSpec,
        body_names: Sequence[str],
        field_name: str,
    ) -> torch.Tensor:
        """Resolve a per-body float spec to a length-``num_bodies`` tensor.

        ``mocap_base`` (mjlab fixed-base wrapper) is skipped and left at 0.
        """
        required = _hydro_required_bodies(body_names)
        if isinstance(spec, Mapping):
            indices, matched_names, values = string_utils.resolve_matching_names_values(
                dict(spec), required, preserve_order=True
            )
            missing = [name for name in required if name not in matched_names]
            assert not missing, (
                f"HydrodynamicsCfg.{field_name} must specify every body; missing: {missing}. "
                f"Bodies: {required}"
            )
            assert len(matched_names) == len(required), (
                f"HydrodynamicsCfg.{field_name} matched {len(matched_names)} bodies, "
                f"expected {len(required)} ({required})"
            )
            out = torch.zeros(len(body_names), dtype=torch.float32)
            name_to_idx = {name: i for i, name in enumerate(body_names)}
            for req_i, value in zip(indices, values):
                out[name_to_idx[required[req_i]]] = float(value)
            return out

        if isinstance(spec, (str, bytes)):
            raise TypeError(
                f"HydrodynamicsCfg.{field_name} must be a sequence of floats or a "
                f"str-to-float mapping; got {type(spec).__name__}"
            )

        try:
            values = list(spec)
        except TypeError as exc:
            raise TypeError(
                f"HydrodynamicsCfg.{field_name} must be a sequence of floats or a "
                f"str-to-float mapping; got {type(spec).__name__}"
            ) from exc

        assert len(values) == len(required), (
            f"HydrodynamicsCfg.{field_name} list length {len(values)} != "
            f"num_bodies {len(required)} ({required})"
        )
        out = torch.zeros(len(body_names), dtype=torch.float32)
        name_to_idx = {name: i for i, name in enumerate(body_names)}
        for name, value in zip(required, values):
            out[name_to_idx[name]] = float(value)
        return out

    def _resolve_body_damping(
        self,
        spec: BodyDampingSpec,
        body_names: Sequence[str],
        field_name: str,
    ) -> torch.Tensor:
        """Resolve per-body damping to a ``(num_bodies, 6)`` tensor.

        ``mocap_base`` (mjlab fixed-base wrapper) is skipped and left at 0.
        """
        required = _hydro_required_bodies(body_names)
        if isinstance(spec, Mapping):
            indices, matched_names, values = string_utils.resolve_matching_names_values(
                dict(spec), required, preserve_order=True
            )
            missing = [name for name in required if name not in matched_names]
            assert not missing, (
                f"HydrodynamicsCfg.{field_name} must specify every body; missing: {missing}. "
                f"Bodies: {required}"
            )
            assert len(matched_names) == len(required), (
                f"HydrodynamicsCfg.{field_name} matched {len(matched_names)} bodies, "
                f"expected {len(required)} ({required})"
            )
            out = torch.zeros(len(body_names), 6, dtype=torch.float32)
            name_to_idx = {name: i for i, name in enumerate(body_names)}
            for req_i, value in zip(indices, values):
                out[name_to_idx[required[req_i]]] = torch.tensor(
                    _expand_damping_coeffs(value), dtype=torch.float32
                )
            return out

        if isinstance(spec, (str, bytes)):
            raise TypeError(
                f"HydrodynamicsCfg.{field_name} must be a sequence or str-to-value "
                f"mapping; got {type(spec).__name__}"
            )

        try:
            values = list(spec)
        except TypeError as exc:
            raise TypeError(
                f"HydrodynamicsCfg.{field_name} must be a sequence or str-to-value "
                f"mapping; got {type(spec).__name__}"
            ) from exc

        assert len(values) == len(required), (
            f"HydrodynamicsCfg.{field_name} list length {len(values)} != "
            f"num_bodies {len(required)} ({required})"
        )
        out = torch.zeros(len(body_names), 6, dtype=torch.float32)
        name_to_idx = {name: i for i, name in enumerate(body_names)}
        for name, value in zip(required, values):
            out[name_to_idx[name]] = torch.tensor(
                _expand_damping_coeffs(value), dtype=torch.float32
            )
        return out

    def _initialize(self, robot: "Articulation", env: "_EnvBase"):
        self.robot = robot
        self.env = env
        self.dt = self.env.sim.get_physics_dt()

        self.body_names = list(self.robot.body_names)
        # Prefer named base_link; fall back to first non-mocap body (Isaac: index 0).
        if "base_link" in self.body_names:
            self._base_body_id = self.body_names.index("base_link")
        else:
            self._base_body_id = next(
                i
                for i, name in enumerate(self.body_names)
                if name not in _HYDRO_SKIP_BODIES
            )
        body_volumes = self._resolve_body_floats(
            self.cfg.volume, self.body_names, "volume"
        ).to(self.device)
        linear_damping = self._resolve_body_damping(
            self.cfg.linear_damping, self.body_names, "linear_damping"
        ).to(self.device)
        quadratic_damping = self._resolve_body_damping(
            self.cfg.quadratic_damping, self.body_names, "quadratic_damping"
        ).to(self.device)

        # Find the rotor bodies once and keep this order as canonical rotor order.
        rotor_indices, rotor_names = self.robot.find_bodies("rotor_.*")
        self.rotor_names = rotor_names
        self.rotor_indices = torch.tensor(rotor_indices, device=self.device, dtype=torch.long)

        time_ids, _, time_values = string_utils.resolve_matching_names_values(
            self._rotor_time_constants, self.rotor_names
        )
        force_ids, _, force_values = string_utils.resolve_matching_names_values(
            self._rotor_force_constants, self.rotor_names
        )
        rotor_time_constants_tensor = torch.zeros(
            self.num_rotors, device=self.device, dtype=torch.float32
        )
        rotor_force_constants_tensor = torch.zeros(
            self.num_rotors, device=self.device, dtype=torch.float32
        )
        rotor_time_constants_tensor[time_ids] = torch.tensor(
            time_values, device=self.device, dtype=torch.float32
        )
        rotor_force_constants_tensor[force_ids] = torch.tensor(
            force_values, device=self.device, dtype=torch.float32
        )

        added_mass_matrix = torch.diag(
            torch.tensor(self.cfg.added_mass, device=self.device)
        ).expand(self.num_envs, -1, -1).clone()

        self.data = UnderwaterRobotData(
            added_mass_matrix=added_mass_matrix,
            linear_damping=linear_damping.unsqueeze(0).expand(self.num_envs, -1, -1).clone(),
            quadratic_damping=quadratic_damping.unsqueeze(0).expand(self.num_envs, -1, -1).clone(),
            volume=body_volumes.unsqueeze(0).expand(self.num_envs, -1).clone(),
            coBM=torch.full((self.num_envs,), self.cfg.coBM, device=self.device),
            prev_body_vels=torch.zeros(self.num_envs, 6, device=self.device),
            prev_body_acc=torch.zeros(self.num_envs, 6, device=self.device),
            flow_vels=torch.zeros(self.num_envs, 6, device=self.device),
            max_flow_vel=torch.zeros(self.num_envs, 6, device=self.device),
            flow_noise_scale=torch.zeros(self.num_envs, 6, device=self.device),
            throttle_cmd=torch.zeros(self.num_envs, self.num_rotors, device=self.device),
            throttle=torch.zeros(self.num_envs, self.num_rotors, device=self.device),
            time_constants=rotor_time_constants_tensor.expand(self.num_envs, -1).clone(),
            force_constants=rotor_force_constants_tensor.expand(self.num_envs, -1).clone(),
            rpm=torch.zeros(self.num_envs, self.num_rotors, device=self.device),
            thrusts_b=torch.zeros(self.num_envs, self.num_rotors, 3, device=self.device),
            body_acc=torch.zeros(self.num_envs, 6, device=self.device),
            damping=torch.zeros(self.num_envs, self.num_bodies, 6, device=self.device),
            added_mass=torch.zeros(self.num_envs, 6, device=self.device),
            coriolis=torch.zeros(self.num_envs, 6, device=self.device),
            buoyancy=torch.zeros(self.num_envs, 6, device=self.device),
            hydro=torch.zeros(self.num_envs, 6, device=self.device),
            hydro_forces_b=torch.zeros(self.num_envs, 3, device=self.device),
            hydro_torques_b=torch.zeros(self.num_envs, 3, device=self.device),
        )
        self._hydro_axis_sign = torch.tensor(
            [1.0, -1.0, -1.0, 1.0, -1.0, -1.0], device=self.device
        )

        # Keep underwater terms in a dedicated namespace to avoid polluting
        # IsaacLab's default articulation data fields.
        self.robot.data_underwater = self.data

    @property
    def num_rotors(self) -> int:
        return len(self.rotor_names)

    def set_flow_velocities(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        max_flow_velocity: Sequence[float],
        flow_velocity_gaussian_noise: Sequence[float],
    ) -> None:
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.data.max_flow_vel[env_ids] = torch.tensor(
            max_flow_velocity, device=self.device, dtype=torch.float32
        )
        self.data.flow_noise_scale[env_ids] = torch.tensor(
            flow_velocity_gaussian_noise, device=self.device, dtype=torch.float32
        )
        self._flow_noise_enabled |= any(
            float(scale) != 0.0 for scale in flow_velocity_gaussian_noise
        )

    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase) -> None:
        self.data.prev_body_vels[env_ids] = 0.0
        self.data.prev_body_acc[env_ids] = 0.0
        self.data.flow_vels[env_ids] = (
            torch.rand_like(self.data.flow_vels[env_ids]) * self.data.max_flow_vel[env_ids]
        )

    def pre_step(self, substep: int):
        self.write_data_to_sim()

    def write_data_to_sim(self):
        # This method will be called by the env before the simulation step.
        # It will be used to update the underwater robot data.
        data = self.robot.data
        with ScopedTimer("underwater.root_flow"):
            root_link_quat_w = data.root_link_quat_w
            if self.env.backend == "isaac":
                # Rotate the COM world twist together with flow below instead
                # of invoking the two derived body-frame properties.
                root_lin_vel_w = data.root_com_lin_vel_w
                root_ang_vel_w = data.root_com_ang_vel_w
            else:
                root_lin_vel_w = data.root_link_lin_vel_w
                root_ang_vel_w = data.root_link_ang_vel_w

            hydro_twist_b, hydro_acc_b, flow_twist_w = _compute_root_flow(
                root_lin_vel_w,
                root_ang_vel_w,
                root_link_quat_w,
                self.data.flow_vels,
                self.data.flow_noise_scale,
                self.data.prev_body_vels,
                self.data.prev_body_acc,
                self._hydro_axis_sign,
                self.dt,
                self.cfg.acc_filter_alpha,
                self._flow_noise_enabled,
            )
            flow_lin_w = flow_twist_w[..., :3]
            flow_ang_w = flow_twist_w[..., 3:]
            self.data.prev_body_vels.copy_(hydro_twist_b)
            self.data.prev_body_acc.copy_(hydro_acc_b)

        with ScopedTimer("underwater.added_mass_coriolis"):
            # Added mass + Coriolis remain vehicle-level on the base twist.
            added_mass_wrench_b = (
                self.data.added_mass_matrix @ hydro_acc_b.unsqueeze(-1)
            ).squeeze(-1)
            added_mass_momentum_b = (
                self.data.added_mass_matrix @ hydro_twist_b.unsqueeze(-1)
            ).squeeze(-1)
            coriolis_wrench_b = torch.zeros(self.num_envs, 6, device=self.device)
            coriolis_wrench_b[:, 0:3] = -torch.cross(
                added_mass_momentum_b[:, 0:3], hydro_twist_b[:, 3:6], dim=-1
            )
            coriolis_wrench_b[:, 3:6] = -(
                torch.cross(
                    added_mass_momentum_b[:, 0:3], hydro_twist_b[:, 0:3], dim=-1
                )
                + torch.cross(
                    added_mass_momentum_b[:, 3:6], hydro_twist_b[:, 3:6], dim=-1
                )
            )

        with ScopedTimer("underwater.buoyancy"):
            # Per-body buoyancy in each body's local frame (same hydro orientation
            # convention as the root). coBM moment arm applies only on the base body.
            body_quat_w = data.body_link_quat_w
            buoyancy_force = (
                self.cfg.water_density * self.cfg.gravity * self.data.volume
            )
            buoyancy_forces_b, buoyancy_torques_b = self._compute_buoyancy(
                body_quat_w,
                buoyancy_force,
                self.data.coBM,
                self._base_body_id,
            )

        with ScopedTimer("underwater.body_damping"):
            # Per-body damping on each link's flow-relative twist.
            with ScopedTimer("underwater.body_state"):
                if self.env.backend == "isaac":
                    # `data.body_link_lin_vel_w` and `data.body_link_ang_vel_w` are slow for isaac, use
                    # `data.body_com_lin_vel_w` and `data.body_com_ang_vel_w` instead.
                    body_lin_vel_w = data.body_com_lin_vel_w
                    body_ang_vel_w = data.body_com_ang_vel_w
                else:
                    body_lin_vel_w = data.body_link_lin_vel_w
                    body_ang_vel_w = data.body_link_ang_vel_w

            with ScopedTimer("underwater.body_damping_math"):
                body_lin_vel_b = quat_rotate_inverse(
                    body_quat_w.reshape(-1, 4),
                    (body_lin_vel_w - flow_lin_w.unsqueeze(1)).reshape(-1, 3),
                ).reshape(self.num_envs, self.num_bodies, 3)
                body_ang_vel_b = quat_rotate_inverse(
                    body_quat_w.reshape(-1, 4),
                    (body_ang_vel_w - flow_ang_w.unsqueeze(1)).reshape(-1, 3),
                ).reshape(self.num_envs, self.num_bodies, 3)
                body_hydro_twist_b = torch.cat(
                    [body_lin_vel_b, body_ang_vel_b], dim=-1
                )
                body_hydro_twist_b.mul_(self._hydro_axis_sign)

                # Damping coefficients are diagonal, so applying the damping
                # matrix is equivalent to an elementwise product.
                damping_wrench_b = (
                    self.data.linear_damping
                    + self.data.quadratic_damping * torch.abs(body_hydro_twist_b)
                ) * body_hydro_twist_b

                hydro_wrench_b = -(added_mass_wrench_b + coriolis_wrench_b)
                hydro_wrench_b.mul_(self._hydro_axis_sign)
                damping_wrench_b.mul_(self._hydro_axis_sign)

        with ScopedTimer("underwater.propulsion"):
            target_throttle = torch.clamp(self.data.throttle_cmd, -1.0, 1.0)
            alpha_rotor = torch.exp(-self.dt / self.data.time_constants)
            self.data.throttle.copy_(
                alpha_rotor * self.data.throttle
                + (1.0 - alpha_rotor) * target_throttle
            )
            target_rpm = torch.where(
                self.data.throttle > 0.075,
                3.6599e3 * self.data.throttle + 3.4521e2,
                torch.where(
                    self.data.throttle < -0.075,
                    3.4944e3 * self.data.throttle - 4.3350e2,
                    0.0,
                ),
            )
            self.data.rpm.copy_(torch.clamp(target_rpm, -3900.0, 3900.0))
            rotor_thrust_force_x = (
                self.data.force_constants
                / 4.4e-7
                * 9.81
                * torch.where(
                    self.data.rpm > 0,
                    4.7368e-7 * torch.square(self.data.rpm)
                    - 1.9275e-4 * self.data.rpm
                    + 8.4452e-2,
                    -3.8442e-7 * torch.square(self.data.rpm)
                    - 1.6186e-4 * self.data.rpm
                    - 3.9139e-2,
                )
            )
            self.data.thrusts_b.zero_()
            # Thrust is along local +X axis of each rotor body.
            self.data.thrusts_b[..., 0] = rotor_thrust_force_x

        with ScopedTimer("underwater.assemble_wrench"):
            forces_b = buoyancy_forces_b - damping_wrench_b[..., 0:3]
            torques_b = buoyancy_torques_b - damping_wrench_b[..., 3:6]
            forces_b[:, self._base_body_id] += hydro_wrench_b[..., 0:3]
            torques_b[:, self._base_body_id] += hydro_wrench_b[..., 3:6]
            forces_b[:, self.rotor_indices] += self.data.thrusts_b

            self.data.body_acc.copy_(hydro_acc_b)
            self.data.damping.copy_(damping_wrench_b)
            self.data.added_mass.copy_(added_mass_wrench_b)
            self.data.coriolis.copy_(coriolis_wrench_b)
            # Debug buoyancy: base-body wrench (force + coBM torque).
            self.data.buoyancy[:, 0:3] = buoyancy_forces_b[:, self._base_body_id]
            self.data.buoyancy[:, 3:6] = buoyancy_torques_b[:, self._base_body_id]
            self.data.hydro.copy_(hydro_wrench_b)
            self.data.hydro_forces_b.copy_(forces_b[:, self._base_body_id])
            self.data.hydro_torques_b.copy_(torques_b[:, self._base_body_id])

        # Isaac: body-local wrenches via permanent_wrench_composer.
        # mjlab: xfrc_applied is world-frame; rotate body wrenches here.
        with ScopedTimer("underwater.submit_wrench"):
            composer = getattr(self.robot, "permanent_wrench_composer", None)
            if composer is not None:
                composer.set_forces_and_torques(
                    forces_b,
                    torques_b,
                    is_global=False,
                )
            else:
                quat_flat = body_quat_w.reshape(-1, 4)
                forces_w = quat_rotate(
                    quat_flat, forces_b.reshape(-1, 3)
                ).reshape_as(forces_b)
                torques_w = quat_rotate(
                    quat_flat, torques_b.reshape(-1, 3)
                ).reshape_as(torques_b)
                self.robot.write_external_wrench_to_sim(forces_w, torques_w)

    def debug_draw(self):
        if self.env.backend == "isaac":
            rotor_pos_w = self.robot.data.body_link_pos_w[:, self.rotor_indices]
            rotor_quat_w = self.robot.data.body_link_quat_w[:, self.rotor_indices]
            v = torch.tensor([[[1.0, 0.0, 0.0]]], device=self.device)
            thrust_w = quat_rotate(rotor_quat_w, v)
            self.env.scene.draw_vector(
                rotor_pos_w.reshape(-1, 3),
                thrust_w.reshape(-1, 3),
                color=(0.2, 0.8, 1.0, 1.0),
            )
