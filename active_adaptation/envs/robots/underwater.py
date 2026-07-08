"""
Backend-agnostic underwater robot wrapper around IsaacLab's Articulation or Mjlab's Entity.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TYPE_CHECKING

import torch
import yaml

from active_adaptation.utils.math import euler_from_quat, quat_rotate_inverse

if TYPE_CHECKING: # DO NOT MODIFY
    # for the editor to work
    from isaaclab.assets import Articulation


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
    base_body_name: str = "base_link"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HydrodynamicsCfg":
        with open(path, "r") as f:
            params = yaml.safe_load(f)
        hydro_coef = params["hydro_coef"]
        return cls(
            volume=params["volume"],
            coBM=params["coBM"],
            added_mass=tuple(hydro_coef["added_mass"]),
            linear_damping=tuple(hydro_coef["linear_damping"]),
            quadratic_damping=tuple(hydro_coef["quadratic_damping"]),
        )


class UnderwaterRobot:
    def __init__(
        self,
        robot: Articulation,
        cfg: HydrodynamicsCfg,
        dt: float,
    ):
        self.robot = robot
        self.cfg = cfg
        self.dt = dt
        self.device = robot.device
        self.num_envs = robot.num_instances

        body_ids, _ = robot.find_bodies(cfg.base_body_name)
        if not body_ids:
            raise ValueError(f"Body '{cfg.base_body_name}' not found on {robot}.")
        self._base_body_id = body_ids[0]

        hydro_coef = cfg
        # TODO: move to a data class to respect isaaclab and mjlab's robot.data API?
        self.added_mass_matrix = torch.diag(
            torch.tensor(hydro_coef.added_mass, device=self.device, dtype=torch.float32)
        ).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.linear_damping_matrix = torch.diag(
            torch.tensor(hydro_coef.linear_damping, device=self.device, dtype=torch.float32)
        ).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.quadratic_damping_matrix = torch.diag(
            torch.tensor(hydro_coef.quadratic_damping, device=self.device, dtype=torch.float32)
        ).unsqueeze(0).repeat(self.num_envs, 1, 1)

        self.volume = torch.full((self.num_envs,), cfg.volume, device=self.device)
        self.coBM = torch.full((self.num_envs,), cfg.coBM, device=self.device)
        self.prev_body_vels = torch.zeros(self.num_envs, 6, device=self.device)
        self.prev_body_acc = torch.zeros(self.num_envs, 6, device=self.device)
        self.flow_vels = torch.zeros(self.num_envs, 6, device=self.device)
        self.max_flow_vel = torch.zeros(self.num_envs, 6, device=self.device)
        self.flow_noise_scale = torch.zeros(self.num_envs, 6, device=self.device)

        self._robot_write_data_to_sim = robot.write_data_to_sim
        robot.write_data_to_sim = self.write_data_to_sim

    def set_flow_velocities(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        max_flow_velocity: Sequence[float],
        flow_velocity_gaussian_noise: Sequence[float],
    ) -> None:
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.max_flow_vel[env_ids] = torch.tensor(
            max_flow_velocity, device=self.device, dtype=torch.float32
        )
        self.flow_noise_scale[env_ids] = torch.tensor(
            flow_velocity_gaussian_noise, device=self.device, dtype=torch.float32
        )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self.prev_body_vels[env_ids] = 0.0
        self.prev_body_acc[env_ids] = 0.0
        self.flow_vels[env_ids] = (
            torch.rand_like(self.flow_vels[env_ids]) * self.max_flow_vel[env_ids]
        )

    def write_data_to_sim(self):
        data = self.robot.data
        rot = data.root_link_quat_w
        body_vels = torch.cat(
            [data.root_link_lin_vel_b, data.root_link_ang_vel_b],
            dim=-1,
        )
        body_rpy = euler_from_quat(rot)

        flow_vels_w = self.flow_vels + torch.rand_like(self.flow_vels) * self.flow_noise_scale
        flow_vels_b = torch.cat(
            [
                quat_rotate_inverse(rot, flow_vels_w[..., :3]),
                quat_rotate_inverse(rot, flow_vels_w[..., 3:]),
            ],
            dim=-1,
        )
        body_vels = body_vels - flow_vels_b
        body_vels[..., [1, 2, 4, 5]] *= -1
        body_rpy = body_rpy.clone()
        body_rpy[..., [1, 2]] *= -1

        alpha = self.cfg.acc_filter_alpha
        acc = (body_vels - self.prev_body_vels) / self.dt
        body_acc = (1.0 - alpha) * self.prev_body_acc + alpha * acc
        self.prev_body_vels = body_vels.clone()
        self.prev_body_acc = body_acc.clone()

        maintained_body_vels = torch.diag_embed(body_vels)
        maintained_body_vels[:, 1, 5] = body_vels[:, 5]
        maintained_body_vels[:, 2, 4] = body_vels[:, 4]
        maintained_body_vels[:, 4, 2] = body_vels[:, 2]
        maintained_body_vels[:, 5, 1] = body_vels[:, 1]
        damping_matrix = self.linear_damping_matrix + self.quadratic_damping_matrix * torch.abs(
            maintained_body_vels
        )
        damping = (damping_matrix @ body_vels.unsqueeze(-1)).squeeze(-1)

        added_mass = (self.added_mass_matrix @ body_acc.unsqueeze(-1)).squeeze(-1)

        ab = (self.added_mass_matrix @ body_vels.unsqueeze(-1)).squeeze(-1)
        coriolis = torch.zeros(self.num_envs, 6, device=self.device)
        coriolis[:, 0:3] = -torch.cross(ab[:, 0:3], body_vels[:, 3:6], dim=-1)
        coriolis[:, 3:6] = -(
            torch.cross(ab[:, 0:3], body_vels[:, 0:3], dim=-1)
            + torch.cross(ab[:, 3:6], body_vels[:, 3:6], dim=-1)
        )

        buoyancy = torch.zeros(self.num_envs, 6, device=self.device)
        buoyancy_force = self.cfg.water_density * self.cfg.gravity * self.volume
        buoyancy[:, 0] = buoyancy_force * torch.sin(body_rpy[:, 1])
        buoyancy[:, 1] = -buoyancy_force * torch.sin(body_rpy[:, 0]) * torch.cos(body_rpy[:, 1])
        buoyancy[:, 2] = -buoyancy_force * torch.cos(body_rpy[:, 0]) * torch.cos(body_rpy[:, 1])
        buoyancy[:, 3] = -self.coBM * buoyancy_force * torch.cos(body_rpy[:, 1]) * torch.sin(body_rpy[:, 0])
        buoyancy[:, 4] = -self.coBM * buoyancy_force * torch.sin(body_rpy[:, 1])

        hydro = -(added_mass + coriolis + damping)
        hydro[:, [1, 2, 4, 5]] *= -1
        buoyancy[:, [1, 2, 4, 5]] *= -1

        forces = (hydro[..., 0:3] + buoyancy[..., 0:3]).unsqueeze(1)
        torques = (hydro[..., 3:6] + buoyancy[..., 3:6]).unsqueeze(1)

        self.robot.set_external_force_and_torque(
            forces,
            torques,
            body_ids=[self._base_body_id],
        )
        self._robot_write_data_to_sim()
