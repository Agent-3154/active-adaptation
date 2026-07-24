"""
Backend-agnostic underwater robot wrapper around IsaacLab's Articulation or Mjlab's Entity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, TYPE_CHECKING

import torch
from tensordict import TensorDictBase

from active_adaptation.utils.math import euler_from_quat, quat_rotate, quat_rotate_inverse
try:
    import isaaclab.utils.string as string_utils
except ModuleNotFoundError:
    from mjlab.utils.lab_api import string as string_utils

if TYPE_CHECKING: # DO NOT MODIFY
    # for the editor to work
    from isaaclab.assets import Articulation
    from active_adaptation.envs.env_base import _EnvBase


@dataclass
class HydrodynamicsCfg:
    volume: float
    coBM: float
    added_mass: tuple[float, float, float, float, float, float]
    linear_damping: tuple[float, float, float, float, float, float]
    quadratic_damping: tuple[float, float, float, float, float, float]
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
    linear_damping_matrix: torch.Tensor
    quadratic_damping_matrix: torch.Tensor
    volume: torch.Tensor
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

    # Per-step decomposed hydrodynamics terms (all in body frame).
    body_acc: torch.Tensor
    damping: torch.Tensor
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
        self.wrench_indices = None
        self.rotor_indices = None
        self.data = None

        # Base body is assumed to be the root body in IsaacLab articulations.
        self._base_body_id = 0
        if robot is not None and env is not None:
            self._initialize(robot=robot, env=env)
        elif robot is not None or env is not None:
            raise ValueError("Both 'robot' and 'env' must be provided together.")

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def device(self) -> torch.device:
        return self.robot.device

    def _initialize(self, robot: "Articulation", env: "_EnvBase"):
        self.robot = robot
        self.env = env
        self.dt = self.env.sim.get_physics_dt()

        # Find the rotor bodies once and keep this order as canonical rotor order.
        rotor_indices, rotor_names = self.robot.find_bodies("rotor_.*")
        self.rotor_names = rotor_names
        self.wrench_indices = torch.tensor(
            [self._base_body_id, *rotor_indices], device=self.device, dtype=torch.long
        )
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

        hydro_coef = self.cfg
        added_mass_matrix = torch.diag(
            torch.tensor(hydro_coef.added_mass, device=self.device)
        ).expand(self.num_envs, -1, -1).clone()
        linear_damping_matrix = torch.diag(
            torch.tensor(hydro_coef.linear_damping, device=self.device)
        ).expand(self.num_envs, -1, -1).clone()
        quadratic_damping_matrix = torch.diag(
            torch.tensor(hydro_coef.quadratic_damping, device=self.device)
        ).expand(self.num_envs, -1, -1).clone()

        self.data = UnderwaterRobotData(
            added_mass_matrix=added_mass_matrix,
            linear_damping_matrix=linear_damping_matrix,
            quadratic_damping_matrix=quadratic_damping_matrix,
            volume=torch.full((self.num_envs,), self.cfg.volume, device=self.device),
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
            damping=torch.zeros(self.num_envs, 6, device=self.device),
            added_mass=torch.zeros(self.num_envs, 6, device=self.device),
            coriolis=torch.zeros(self.num_envs, 6, device=self.device),
            buoyancy=torch.zeros(self.num_envs, 6, device=self.device),
            hydro=torch.zeros(self.num_envs, 6, device=self.device),
            hydro_forces_b=torch.zeros(self.num_envs, 3, device=self.device),
            hydro_torques_b=torch.zeros(self.num_envs, 3, device=self.device),
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
        root_link_quat_w = data.root_link_quat_w
        root_link_twist_b = torch.cat(
            [data.root_link_lin_vel_b, data.root_link_ang_vel_b],
            dim=-1,
        )
        root_link_rpy_w = euler_from_quat(root_link_quat_w)

        flow_twist_w = self.data.flow_vels + torch.rand_like(self.data.flow_vels) * self.data.flow_noise_scale
        flow_twist_b = torch.cat(
            [
                quat_rotate_inverse(root_link_quat_w, flow_twist_w[..., :3]),
                quat_rotate_inverse(root_link_quat_w, flow_twist_w[..., 3:]),
            ],
            dim=-1,
        )
        # Relative body twist after subtracting ocean current, then converted
        # to the hydrodynamics sign convention used by the fitted coefficients.
        hydro_twist_b = root_link_twist_b - flow_twist_b
        hydro_twist_b[..., [1, 2, 4, 5]] *= -1
        hydro_rpy = root_link_rpy_w.clone()
        hydro_rpy[..., [1, 2]] *= -1

        alpha = self.cfg.acc_filter_alpha
        hydro_acc_b = (hydro_twist_b - self.data.prev_body_vels) / self.dt
        hydro_acc_b = (1.0 - alpha) * self.data.prev_body_acc + alpha * hydro_acc_b
        self.data.prev_body_vels.copy_(hydro_twist_b)
        self.data.prev_body_acc.copy_(hydro_acc_b)

        hydro_twist_matrix_b = torch.diag_embed(hydro_twist_b)
        hydro_twist_matrix_b[:, 1, 5] = hydro_twist_b[:, 5]
        hydro_twist_matrix_b[:, 2, 4] = hydro_twist_b[:, 4]
        hydro_twist_matrix_b[:, 4, 2] = hydro_twist_b[:, 2]
        hydro_twist_matrix_b[:, 5, 1] = hydro_twist_b[:, 1]
        damping_matrix = self.data.linear_damping_matrix + self.data.quadratic_damping_matrix * torch.abs(
            hydro_twist_matrix_b
        )
        damping_wrench_b = (damping_matrix @ hydro_twist_b.unsqueeze(-1)).squeeze(-1)

        added_mass_wrench_b = (self.data.added_mass_matrix @ hydro_acc_b.unsqueeze(-1)).squeeze(-1)

        added_mass_momentum_b = (self.data.added_mass_matrix @ hydro_twist_b.unsqueeze(-1)).squeeze(-1)
        coriolis_wrench_b = torch.zeros(self.num_envs, 6, device=self.device)
        coriolis_wrench_b[:, 0:3] = -torch.cross(
            added_mass_momentum_b[:, 0:3], hydro_twist_b[:, 3:6], dim=-1
        )
        coriolis_wrench_b[:, 3:6] = -(
            torch.cross(added_mass_momentum_b[:, 0:3], hydro_twist_b[:, 0:3], dim=-1)
            + torch.cross(added_mass_momentum_b[:, 3:6], hydro_twist_b[:, 3:6], dim=-1)
        )

        buoyancy_wrench_b = torch.zeros(self.num_envs, 6, device=self.device)
        buoyancy_force = self.cfg.water_density * self.cfg.gravity * self.data.volume
        buoyancy_wrench_b[:, 0] = buoyancy_force * torch.sin(hydro_rpy[:, 1])
        buoyancy_wrench_b[:, 1] = -buoyancy_force * torch.sin(hydro_rpy[:, 0]) * torch.cos(hydro_rpy[:, 1])
        buoyancy_wrench_b[:, 2] = -buoyancy_force * torch.cos(hydro_rpy[:, 0]) * torch.cos(hydro_rpy[:, 1])
        buoyancy_wrench_b[:, 3] = (
            -self.data.coBM * buoyancy_force * torch.cos(hydro_rpy[:, 1]) * torch.sin(hydro_rpy[:, 0])
        )
        buoyancy_wrench_b[:, 4] = -self.data.coBM * buoyancy_force * torch.sin(hydro_rpy[:, 1])

        hydro_wrench_b = -(added_mass_wrench_b + coriolis_wrench_b + damping_wrench_b)
        hydro_wrench_b[:, [1, 2, 4, 5]] *= -1
        buoyancy_wrench_b[:, [1, 2, 4, 5]] *= -1

        base_link_force_b = (hydro_wrench_b[..., 0:3] + buoyancy_wrench_b[..., 0:3]).unsqueeze(1)
        base_link_torque_b = (hydro_wrench_b[..., 3:6] + buoyancy_wrench_b[..., 3:6]).unsqueeze(1)

        target_throttle = torch.clamp(self.data.throttle_cmd, -1.0, 1.0)
        alpha_rotor = torch.exp(-self.dt / self.data.time_constants)
        self.data.throttle.copy_(
            alpha_rotor * self.data.throttle + (1.0 - alpha_rotor) * target_throttle
        )
        target_rpm = torch.where(
            self.data.throttle > 0.075,
            3.6599e3 * self.data.throttle + 3.4521e2,
            torch.where(
                self.data.throttle < -0.075,
                3.4944e3 * self.data.throttle - 4.3350e2,
                torch.zeros_like(self.data.throttle),
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

        self.data.body_acc.copy_(hydro_acc_b)
        self.data.damping.copy_(damping_wrench_b)
        self.data.added_mass.copy_(added_mass_wrench_b)
        self.data.coriolis.copy_(coriolis_wrench_b)
        self.data.buoyancy.copy_(buoyancy_wrench_b)
        self.data.hydro.copy_(hydro_wrench_b)
        self.data.hydro_forces_b.copy_(base_link_force_b.squeeze(1))
        self.data.hydro_torques_b.copy_(base_link_torque_b.squeeze(1))

        # With `is_global=False`, each entry is interpreted in the local frame of
        # the corresponding body in `self.wrench_indices`: [base_link, rotors...].
        combined_forces_b = torch.cat([base_link_force_b, self.data.thrusts_b], dim=1)
        rotor_torques_b = torch.zeros_like(self.data.thrusts_b)
        combined_torques_b = torch.cat([base_link_torque_b, rotor_torques_b], dim=1)
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            combined_forces_b,
            combined_torques_b,
            body_ids=self.wrench_indices,
            is_global=False,
        )
    
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
