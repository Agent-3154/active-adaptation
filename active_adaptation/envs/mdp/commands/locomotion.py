from math import pi
import torch
import torch.nn.functional as F
import torch.distributions as D
import math
import warp as wp
from typing import Sequence, Tuple
from typing_extensions import override

from active_adaptation.utils.math import (
    quat_from_euler_xyz,
    quat_rotate, 
    quat_rotate_inverse,
    clamp_norm,
    yaw_quat,
    yaw_rotate,
    wrap_to_pi,
    MultiUniform
)
import active_adaptation.utils.symmetry as symmetry_utils
from active_adaptation.envs.mdp.base import Command


class Twist(Command):

    def __init__(
        self,
        env,
        linvel_x_range=(-1.0, 1.0),
        linvel_y_range=(-1.0, 1.0),
        angvel_range=(-1, 1),
        yaw_stiffness_range=(0.5, 0.6),
        use_stiffness_ratio: float = 0.5,
        base_height_range=(0.2, 0.4),
        resample_interval: int = 300,
        resample_prob: float = 0.75,
        stand_prob=0.2,
        target_yaw_range=(0, torch.pi * 2),
        curriculum: bool = False,
    ):
        super().__init__(env)
        self.linvel_x_range = linvel_x_range
        self.linvel_y_range = linvel_y_range
        self.angvel_range = angvel_range
        self.use_stiffness_ratio = use_stiffness_ratio
        self.yaw_stiffness_range = yaw_stiffness_range
        self.base_height_range = base_height_range
        self.resample_interval = resample_interval
        self.resample_prob = resample_prob
        self.stand_prob = stand_prob
        self.curriculum = curriculum and self.env.backend == "isaac"

        if self.curriculum:
            self.terrain = self.env.scene.terrain
            assert self.terrain.cfg.terrain_type == "generator", "Curriculum is only supported for generator terrain"
            assert self.terrain.cfg.terrain_generator.curriculum, "Curriculum is not enabled for the terrain"

        with torch.device(self.device):
            if all(isinstance(r, Sequence) for r in target_yaw_range):
                self.target_yaw_dist = MultiUniform(torch.tensor(target_yaw_range))
            else:
                self.target_yaw_dist = D.Uniform(*torch.tensor(target_yaw_range))

            self.target_yaw = torch.zeros(self.num_envs, 1)
            self.yaw_stiffness = torch.zeros(self.num_envs, 1)
            self.use_stiffness = torch.zeros(self.num_envs, 1, dtype=bool)
            self.fixed_yaw_speed = torch.zeros(self.num_envs, 1)

            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)

            self.command_speed = torch.zeros(self.num_envs, 1)
            self.next_command_linvel = torch.zeros(self.num_envs, 3)
            self.cmd_linvel_b = torch.zeros(self.num_envs, 3)
            self.cmd_linvel_w = torch.zeros(self.num_envs, 3)
            self.cmd_yawvel_b = torch.zeros(self.num_envs, 1)
            self.cmd_base_height = torch.zeros(self.num_envs, 1)

            self.distance_commanded = torch.zeros(self.num_envs, 1)
            self.distance_traveled = torch.zeros(self.num_envs, 1)

            self._cum_error = torch.zeros(self.num_envs, 2)
            self._cum_linvel_error = self._cum_error[:, 0].unsqueeze(1)
            self._cum_angvel_error = self._cum_error[:, 1].unsqueeze(1)

        if self.teleop:
            self.key_mappings_pos = {
                "W": torch.tensor(
                    [self.linvel_x_range[1], 0.0, 0.0], device=self.device
                ),
                "S": torch.tensor(
                    [self.linvel_x_range[0], 0.0, 0.0], device=self.device
                ),
                "A": torch.tensor(
                    [0.0, self.linvel_y_range[1], 0.0], device=self.device
                ),
                "D": torch.tensor(
                    [0.0, self.linvel_y_range[0], 0.0], device=self.device
                ),
            }
        
        if self.env.sim.has_gui():
            if self.env.backend == "mjlab":
                from active_adaptation.viewer import MjLabViewer
                self.viewer: MjLabViewer = self.env.sim.viewer
                self.axes_handle = self.viewer.add_batched_axes("target_yaw")
                self.lines_handle = self.viewer.add_line_segments("cmd_linvel_w", (1., 0., 0.))
                self.lines_handle.line_width = 2.0
    
    @property
    def command(self):
        return torch.cat([
            self.cmd_linvel_b[:, :2],
            self.cmd_yawvel_b.reshape(self.num_envs, 1),
            self.cmd_base_height.reshape(self.num_envs, 1),
        ], dim=-1)

    @override
    def reset(self, env_ids):
        self.next_command_linvel[env_ids] = 0.0
        self.cmd_linvel_b[env_ids] = 0.0
        self.target_yaw[env_ids] = self.asset.data.heading_w[env_ids, None]
        self.cmd_yawvel_b[env_ids] = 0.0

        self._cum_linvel_error[env_ids] = 0.0
        self._cum_angvel_error[env_ids] = 0.0
        self.is_standing_env[env_ids] = True
    
    @override
    def sample_init(self, env_ids):
        if self.curriculum and self.env.episode_count > 1: # and self.env.training:
            distance_traveled = self.distance_traveled[env_ids]
            distance_commanded = self.distance_commanded[env_ids].clamp_min(1.0)
            move_up = distance_traveled > distance_commanded * 0.8
            move_down = distance_traveled < distance_commanded * 0.4
            move_up = move_up & ~move_down
            self.terrain.update_env_origins(env_ids, move_up.squeeze(-1), move_down.squeeze(-1))
            self._origins = self.terrain.env_origins.clone()
            self.env.extra["curriculum/terrain_level"] = self.terrain.terrain_levels.float().mean()
        self.env.extra["curriculum/distance_commanded"] = self.distance_commanded.mean()
        self.env.extra["curriculum/distance_traveled"] = self.distance_traveled.mean()
        self.distance_commanded[env_ids] = 0.0
        self.distance_traveled[env_ids] = 0.0
        return super().sample_init(env_ids)

    @override
    def update(self):
        self.body_heading_w = self.asset.data.heading_w.unsqueeze(1)
        self.lin_vel_w = self.asset.data.root_com_lin_vel_w
        self.ang_vel_w = self.asset.data.root_com_ang_vel_w
        self.quat_w = self.asset.data.root_link_quat_w

        # this is used for terminating episodes where the robot is inactive due to whatever reason
        linvel_diff = self.lin_vel_w[:, :2] - self.cmd_linvel_w[:, :2]
        linvel_error = linvel_diff.norm(dim=-1, keepdim=True)
        angvel_diff = self.cmd_yawvel_b - self.ang_vel_w[:, 2:3]
        angvel_error = angvel_diff.abs()

        self._cum_linvel_error.mul_(0.98).add_(linvel_error * self.env.step_dt)
        self._cum_angvel_error.mul_(0.98).add_(angvel_error * self.env.step_dt)

        max_command_speed = (2.5 - self.cmd_yawvel_b.abs()).clamp(0.0)
        self.cmd_linvel_b.lerp_(self.next_command_linvel, 0.1)
        self.cmd_linvel_b = clamp_norm(self.cmd_linvel_b, max=max_command_speed)
        self.command_speed = self.cmd_linvel_b.norm(dim=-1, keepdim=True)
    
        self.current_speed = self.lin_vel_w.norm(dim=-1, keepdim=True)
        self.distance_commanded = self.distance_commanded + self.command_speed * self.env.step_dt
        self.distance_traveled = self.distance_traveled + self.current_speed * self.env.step_dt

        interval_reached = (self.env.episode_length_buf - 20) % self.resample_interval == 0
        resample_vel = interval_reached & (
            self.with_prob(self.num_envs, self.resample_prob)
            | self.is_standing_env.squeeze(1)
        )
        resample_yaw = interval_reached & (
            self.with_prob(self.num_envs, self.resample_prob)
            | self.is_standing_env.squeeze(1)
        )
        self.sample_vel_command(resample_vel.nonzero().squeeze(-1))
        self.sample_yaw_command(resample_yaw.nonzero().squeeze(-1))

        yaw_diff = wrap_to_pi(self.target_yaw - self.body_heading_w).reshape(self.num_envs, 1)
        cmd_yawvel_b = torch.clamp(
            self.yaw_stiffness * yaw_diff,
            min=self.angvel_range[0],
            max=self.angvel_range[1],
        ).reshape(self.num_envs, 1)

        self.cmd_yawvel_b = torch.where(
            self.use_stiffness,
            cmd_yawvel_b,
            self.fixed_yaw_speed
        ).reshape(self.num_envs, 1)

        self.cmd_linvel_w = quat_rotate(yaw_quat(self.quat_w), self.cmd_linvel_b)
        self.is_standing_env = (self.command_speed < 0.1) & (self.cmd_yawvel_b.abs() < 0.1)

    def sample_vel_command(self, env_ids: torch.Tensor):
        next_command_linvel = torch.zeros(len(env_ids), 3, device=self.device)
        next_command_linvel[:, 0].uniform_(*self.linvel_x_range)
        next_command_linvel[:, 1].uniform_(*self.linvel_y_range)

        speed = next_command_linvel.norm(dim=-1, keepdim=True)
        r = torch.rand(len(env_ids), 1, device=self.device) < self.stand_prob
        valid = ~((speed < 0.10) | r)
        self.next_command_linvel[env_ids] = next_command_linvel * valid
        self.cmd_base_height[env_ids] = sample_uniform((len(env_ids), 1), *self.base_height_range, self.device)

    def sample_yaw_command(self, env_ids: torch.Tensor):
        self.target_yaw[env_ids] = self.target_yaw_dist.sample(env_ids.shape).unsqueeze(1)
        shape = (len(env_ids), 1)
        self.yaw_stiffness[env_ids] = sample_uniform(shape, *self.yaw_stiffness_range, self.device)
        self.use_stiffness[env_ids] = self.with_prob(shape, self.use_stiffness_ratio)
        self.fixed_yaw_speed[env_ids] = sample_uniform(shape, *self.angvel_range, self.device)

    def with_prob(self, n, p):
        return torch.rand(n, device=self.device) < p
    
    @override
    def debug_draw(self):
        start = self.asset.data.root_link_pos_w + torch.tensor([0.0, 0.0, 0.2], device=self.device)
        yaw_vec = torch.stack(
            [
                self.target_yaw.cos(),
                self.target_yaw.sin(),
                torch.zeros_like(self.target_yaw),
            ],
            1,
        )
        if self.env.backend == "isaac":
            self.env.debug_draw.vector(
                start,
                self.cmd_linvel_w,
                color=(1.0, 1.0, 1.0, 1.0),
            )
            self.env.debug_draw.vector(
                start,
                yaw_vec,
                color=(0.2, 0.2, 1.0, 1.0),
            )
        elif self.env.backend == "mjlab":
            rpy = torch.zeros(self.num_envs, 3)
            rpy[:, 2] = self.target_yaw.cpu()
            self.axes_handle.batched_wxyzs = quat_from_euler_xyz(rpy)
            self.axes_handle.batched_positions = start.cpu()
            self.lines_handle.points = torch.stack([start, start + self.cmd_linvel_w], 1).cpu()
    
    def symmetry_transform(self):
        # left-right symmetry: flip y velocity and yaw velocity
        transform = symmetry_utils.SymmetryTransform(perm=torch.arange(4), signs=[1, -1, -1, 1])
        return transform


class PositionVelocityTracking(Command):
    def __init__(
        self,
        env,
        linvel_x_range: Tuple[float, float] = (-1.0, 1.0),
        linvel_y_range: Tuple[float, float] = (-1.0, 1.0),
        angvel_range: Tuple[float, float] = (-1.0, 1.0),
        resample_interval: int = 300,
        resample_prob: float = 0.75,
        curriculum: bool = False,
    ):
        super().__init__(env)
        self.linvel_x_range = linvel_x_range
        self.linvel_y_range = linvel_y_range
        self.angvel_range = angvel_range
        self.resample_interval = resample_interval
        self.resample_prob = resample_prob
        self.curriculum = curriculum and self.env.backend == "isaac"

        if self.curriculum:
            from isaaclab.terrains import TerrainImporter
            self.terrain: TerrainImporter = self.env.scene.terrain
            assert self.terrain.cfg.terrain_type == "generator", "Curriculum is only supported for generator terrain"
            assert self.terrain.cfg.terrain_generator.curriculum, "Curriculum is not enabled for the terrain"

        with torch.device(self.device):
            self.ref_pos_w = torch.zeros(self.num_envs, 3)
            self.ref_yaw_w = torch.zeros(self.num_envs, 1)
            self.ref_linvel_b_next = torch.zeros(self.num_envs, 3) # intermediate term for smooth transition
            self.ref_linvel_b = torch.zeros(self.num_envs, 3)
            self.ref_yawvel_w = torch.zeros(self.num_envs, 1)
            
            self.ref_linvel_w: torch.Tensor # derived from ref_lin_vel_b and ref_yaw_w
            self.cmd_linvel_w: torch.Tensor
            self.cmd_yawvel_w: torch.Tensor
            self.command_speed: torch.Tensor # derived from cmd_linvel_w

            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)
            self.distance_commanded = torch.zeros(self.num_envs, 1)
            self.distance_traveled = torch.zeros(self.num_envs, 1)
        
        if self.env.sim.has_gui():
            if self.env.backend == "isaac":
                from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg, sim_utils
                self.marker = VisualizationMarkers(
                    VisualizationMarkersCfg(
                        prim_path=f"/Visuals/Command/ref_pos_w",
                        markers={
                            "ref_pos_w": sim_utils.SphereCfg(radius=0.04, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 1.0))),
                        }
                    )
                )
                self.marker.set_visibility(True)
        
        self.update()
    
    @property
    def command(self):
        quat = yaw_quat(self.asset.data.root_link_quat_w)
        return torch.cat([
            quat_rotate_inverse(quat, self.ref_pos_w - self.asset.data.root_link_pos_w)[:, :2], # 2
            quat_rotate_inverse(quat, self.cmd_linvel_w)[:, :2], # 2
            wrap_to_pi(self.ref_yaw_w - self.asset.data.heading_w.reshape(self.num_envs, 1)), # 1
            self.cmd_yawvel_w.reshape(self.num_envs, 1), # 1
        ], dim=-1)
    
    @override
    def symmetry_transform(self):
        # left-right symmetry: flip y velocity and yaw velocity
        transform = symmetry_utils.SymmetryTransform(
            perm=torch.arange(2 + 2 + 1 + 1),
            signs=[1, -1] + [1, -1] + [-1, -1]
        )
        return transform
    
    @override
    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        if self.curriculum and self.env.episode_count > 1: # and self.env.training:
            distance_traveled = self.distance_traveled[env_ids]
            distance_commanded = self.distance_commanded[env_ids].clamp_min(1.0)
            move_up = distance_traveled > distance_commanded * 0.8
            move_down = distance_traveled < distance_commanded * 0.4
            move_up = move_up & ~move_down
            self.terrain.update_env_origins(env_ids, move_up.squeeze(-1), move_down.squeeze(-1))
            self._origins = self.terrain.env_origins.clone()
            self.env.extra["curriculum/terrain_level"] = self.terrain.terrain_levels.float().mean()
        self.env.extra["curriculum/distance_commanded"] = self.distance_commanded.mean()
        self.env.extra["curriculum/distance_traveled"] = self.distance_traveled.mean()
        self.distance_commanded[env_ids] = 0.0
        self.distance_traveled[env_ids] = 0.0
        return super().sample_init(env_ids)

    @override
    def reset(self, env_ids):
        self.ref_pos_w[env_ids] = self.asset.data.root_link_pos_w[env_ids]
        self.ref_yaw_w[env_ids] = self.asset.data.heading_w[env_ids, None]
        self.ref_linvel_b[env_ids] = 0.0
        self.ref_yawvel_w[env_ids] = 0.0
        self.is_standing_env[env_ids] = False
    
    @override
    def update(self):
        self.ref_linvel_b = (
            self.ref_linvel_b 
            + clamp_norm((self.ref_linvel_b_next - self.ref_linvel_b) * 0.1, max=0.1)
        )
        self.ref_linvel_w = yaw_rotate(self.ref_yaw_w, self.ref_linvel_b)
        self.ref_pos_w = self.ref_pos_w + self.ref_linvel_w * self.env.step_dt
        self.ref_yaw_w = self.ref_yaw_w + self.ref_yawvel_w * self.env.step_dt

        self.cmd_linvel_w = (
            (self.ref_pos_w - self.asset.data.root_link_pos_w)
            + self.ref_linvel_w
        )
        self.command_speed = self.cmd_linvel_w[:, :2].norm(dim=-1, keepdim=True)
        self.current_speed = self.asset.data.root_com_lin_vel_w[:, :2].norm(dim=-1, keepdim=True)
        self.distance_commanded = self.distance_commanded + self.command_speed * self.env.step_dt
        self.distance_traveled = self.distance_traveled + self.current_speed * self.env.step_dt

        self.cmd_yawvel_w = (
            wrap_to_pi(self.ref_yaw_w - self.asset.data.heading_w.reshape(self.num_envs, 1))
            + self.ref_yawvel_w
        )
        self.cmd_pos_w = self.ref_pos_w.clone()
        self.cmd_yawvel_b = self.cmd_yawvel_w.clone()

        resample_lin_vel = (
            ((self.env.episode_length_buf - 20) % self.resample_interval == 0)
            & (torch.rand(self.num_envs, device=self.device) < self.resample_prob)
        )
        resample_ids = resample_lin_vel.nonzero().squeeze(-1)
        if len(resample_ids):
            ref_lin_vel_b = torch.zeros(len(resample_ids), 3, device=self.device)
            ref_lin_vel_b[:, 0].uniform_(*self.linvel_x_range)
            ref_lin_vel_b[:, 1].uniform_(*self.linvel_y_range)
            ref_lin_vel_b = torch.where(
                ref_lin_vel_b.norm(dim=-1, keepdim=True) < 0.1,
                0.0,
                ref_lin_vel_b,
            )
            self.ref_linvel_b_next[resample_ids] = ref_lin_vel_b
        
        resample_yaw_vel = (
            ((self.env.episode_length_buf - 20) % self.resample_interval == 0)
            & (torch.rand(self.num_envs, device=self.device) < self.resample_prob)
        )
        resample_ids = resample_yaw_vel.nonzero().squeeze(-1)
        if len(resample_ids):
            ref_yaw_vel_w = torch.zeros(len(resample_ids), 1, device=self.device)
            ref_yaw_vel_w.uniform_(*self.angvel_range)
            self.ref_yawvel_w[resample_ids] = ref_yaw_vel_w

    def debug_draw(self):
        if self.env.backend == "isaac":
            self.marker.visualize(self.ref_pos_w)


def sample_uniform(size, low: float, high: float, device: torch.device = "cpu"):
    return torch.rand(size, device=device) * (high - low) + low


def quat_to_yaw(quat: torch.Tensor):
    q_w, q_x, q_y, q_z = quat.unbind(-1)
    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = torch.atan2(sin_yaw, cos_yaw)
    return yaw % (2 * torch.pi)


