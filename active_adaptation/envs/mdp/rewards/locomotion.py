from math import inf
import torch
import abc
from typing import TYPE_CHECKING, Optional

import isaaclab.utils.string as string_utils
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse, yaw_quat
from active_adaptation.envs.mdp.base import Reward
from active_adaptation.envs.mdp.commands.locomotion import Twist

if TYPE_CHECKING:
    from isaaclab.sensors import ContactSensor
    from isaaclab.assets import Articulation


class survival(Reward):
    def compute(self):
        return torch.ones(self.num_envs, 1, device=self.device)


class linvel_z_l2(Reward):
    def __init__(self, env, weight: float):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene["robot"]

    def compute(self) -> torch.Tensor:
        linvel_z = self.asset.data.root_lin_vel_b[:, 2].unsqueeze(1)
        return -linvel_z.square()


class angvel_xy_l2(Reward):
    def __init__(self, env, weight: float, body_names: str = None):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene["robot"]
        if body_names is not None:
            self.body_ids, self.body_names = self.asset.find_bodies(body_names)
            self.body_ids = torch.tensor(self.body_ids, device=self.device)
        else:
            self.body_ids = None

    def update(self):
        if self.body_ids is not None:
            angvel = quat_rotate_inverse(
                self.asset.data.body_quat_w[:, self.body_ids],
                self.asset.data.body_ang_vel_w[:, self.body_ids]
            )
        else:
            angvel = self.asset.data.root_ang_vel_b
        self.angvel = angvel

    def compute(self) -> torch.Tensor:
        if self.body_ids is not None:
            r = -self.angvel[:, :, :2].square().sum(-1).mean(1)
        else:
            r = -self.angvel[:, :2].square().sum(-1)
        return r.reshape(self.num_envs, 1)


class undesired_contact(Reward):
    def __init__(self, env, body_names, weight: float):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.num_bodies = len(self.body_ids)

    def update(self):
        contact = self.contact_sensor.data.current_contact_time[:, self.body_ids] > 0.0
        self.undesired_contact = - contact.float().sum(1, keepdim=True)
        self.env.discount.mul_((self.undesired_contact / self.num_bodies).exp())

    def compute(self) -> torch.Tensor:
        return self.undesired_contact.reshape(self.num_envs, 1)

    # def debug_draw(self):
    #     self.env.debug_draw.point(
    #         # self.contact_sensor.data.pos_w[:, self.body_ids],
    #         self.asset.data.body_pos_w[:, self.articulation_body_ids],
    #         color=(1., .6, .4, 1.),
    #         size=20,
    #     )


class impact_force_l2(Reward):
    def __init__(self, env, body_names, weight: float):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene["robot"]
        self.default_mass_total = (
            self.asset.root_physx_view.get_masses()[0].sum() * 9.81
        )
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)

        print(f"Penalizing impact forces on {self.body_names}.")

    def compute(self) -> torch.Tensor:
        first_contact = self.contact_sensor.compute_first_contact(self.env.step_dt)[
            :, self.body_ids
        ]
        contact_forces = self.contact_sensor.data.net_forces_w_history.norm(
            dim=-1
        ).mean(1)
        force = contact_forces[:, self.body_ids] / self.default_mass_total
        return -(force.square() * first_contact).sum(1, True)


class linvel_exp(Reward[Twist]):
    def __init__(
        self,
        env,
        weight: float,
        sigma: float = 0.25,
        dim: int = 2,
        gamma: float = 0.0,
    ):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene["robot"]
        self.sigma = sigma
        self.dim = dim
        self.gamma = gamma

        self.linvel_w_sum = torch.zeros(self.num_envs, 3, device=self.device)
        self.count = torch.zeros(self.num_envs, 1, device=self.device)

    def reset(self, env_ids):
        self.linvel_w_sum[env_ids] = 0.0
        self.count[env_ids] = 0.0

    def update(self):
        linvel_w = self.asset.data.root_link_lin_vel_w
        self.linvel_w_sum.mul_(self.gamma).add_(linvel_w)
        self.count.mul_(self.gamma).add_(1.0)

    def compute(self) -> torch.Tensor:
        linvel_w = self.linvel_w_sum / self.count.clamp_min(1.0)
        cmd_linvel_w = self.command_manager.cmd_linvel_w[:, : self.dim]
        linvel_error = (linvel_w[:, : self.dim] - cmd_linvel_w).square().sum(-1, True)
        rew = torch.exp(-linvel_error / self.sigma)
        return rew.reshape(self.num_envs, 1)

    def debug_draw(self):
        # draw smoothed lin vel (purple)
        linvel_w = self.linvel_w_sum / self.count.clamp_min(1.0)
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w
            + torch.tensor([0.0, 0.0, 0.2], device=self.device),
            linvel_w,
            color=(0.8, 0.1, 0.8, 1.0),
        )


class linvel_projection(Reward[Twist]):
    def __init__(
        self,
        env,
        weight: float,
        dim: int = 2,
    ):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene["robot"]
        self.dim = dim

    def compute(self) -> torch.Tensor:
        linvel_w = self.asset.data.root_link_lin_vel_w[:, : self.dim]
        cmd_linvel_w = self.command_manager.cmd_linvel_w[:, : self.dim]
        projection = (linvel_w * cmd_linvel_w).sum(dim=-1, keepdim=True)
        rew = projection.clamp_max(self.command_manager.command_speed)
        return rew.reshape(self.num_envs, 1)


class angvel_z_exp(Reward[Twist]):
    def __init__(
        self,
        env,
        weight: float,
        world_frame: bool = False,
        gamma: float = 0.0,
    ):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene["robot"]
        self.world_frame = world_frame
        self.gamma = gamma
        self.count = torch.zeros(self.num_envs, 1, device=self.device)
        self.angvel_sum = torch.zeros(self.num_envs, 3, device=self.device)

    def update(self):
        if self.world_frame:
            angvel = self.asset.data.root_ang_vel_w
        else:
            angvel = self.asset.data.root_ang_vel_b
        self.angvel_sum.mul_(self.gamma).add_(angvel)
        self.count.mul_(self.gamma).add_(1)

    def compute(self) -> torch.Tensor:
        angvel = self.angvel_sum / self.count.clamp_min(1.0)
        target_angvel = self.command_manager.cmd_yawvel_b
        angvel_error = (target_angvel - angvel[:, 2:3]).square()
        rew = torch.exp(-angvel_error / 0.25)
        return rew.reshape(self.num_envs, 1)


class tracking_lin_vel(Reward):
    def __init__(self, env, weight):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene["robot"]
        self.command_manager = self.env.command_manager
        self.decay = torch.tensor([0.0, 0.5, 0.8], device=self.device)
        self.lin_vel_w = torch.zeros(self.num_envs, 3, len(self.decay), device=self.device)
        self.cnt = torch.zeros(self.num_envs, 1, 3, device=self.device)

    def reset(self, env_ids):
        self.lin_vel_w[env_ids] = 0.0
        self.cnt[env_ids] = 0.0
        
    def update(self):
        self.lin_vel_w.mul_(self.decay).add_(self.asset.data.root_lin_vel_w.unsqueeze(-1))
        self.cnt.mul_(self.decay).add_(1.0)
    
    def compute(self):
        command_linvel = self.command_manager.command_linvel_w[:, :2].unsqueeze(-1)
        error = (command_linvel - (self.lin_vel_w[:, :2] / self.cnt)).square().sum(1)
        return torch.exp(- error.min(dim=1, keepdim=True).values.square() / 0.25)


class tracking_yaw(Reward):
    def __init__(self, env, weight):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene["robot"]
        self.command_manager = self.env.command_manager

    def compute(self):
        yaw_diff = self.command_manager.ref_yaw - self.asset.data.heading_w.unsqueeze(1)
        return torch.exp(- yaw_diff.square())


class angvel_z_exp_shaped(Reward):
    def __init__(
        self, env, weight: float, enabled: bool = True, world_frame: bool = False
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.target_angvel: torch.Tensor = self.env.command_manager.command_angvel
        self.world_frame = world_frame

    def compute(self) -> torch.Tensor:
        if self.world_frame:
            angvel_z = self.asset.data.root_ang_vel_w[:, 2]
        else:
            angvel_z = self.asset.data.root_ang_vel_b[:, 2]
        angvel_error = (self.target_angvel - angvel_z).unsqueeze(1)
        angvel_error = shaped_error(angvel_error)
        r = torch.exp(-angvel_error / 0.25)
        return r


class linacc_z_l2(Reward):
    def __init__(self, env, weight: float):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene["robot"]
        self.prev_linvel_z = torch.zeros(self.env.num_envs, device=self.env.device)
        self.linacc_z = torch.zeros(self.env.num_envs, device=self.env.device)

    def reset(self, env_ids):
        self.prev_linvel_z[env_ids] = 0.0
        self.linacc_z[env_ids] = 0.0

    def update(self):
        self.linacc_z[:] = (
            self.asset.data.root_lin_vel_b[:, 2] - self.prev_linvel_z
        ) / self.env.step_dt
        self.prev_linvel_z[:] = self.asset.data.root_lin_vel_b[:, 2]

    def compute(self) -> torch.Tensor:
        return -self.linacc_z.square().unsqueeze(1)


class feet_air_time(Reward):
    def __init__(self, env, body_names: str, thres: float, weight: float, condition_on_linvel: bool = False):
        super().__init__(env, weight)
        self.thres = thres
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.condition_on_linvel = condition_on_linvel

        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)
        self.reward = torch.zeros(self.num_envs, 1, device=self.env.device)
        self.last_air_time = torch.zeros(
            self.num_envs, len(self.body_ids), device=self.env.device
        )

    def compute(self):
        first_contact = self.contact_sensor.compute_first_contact(self.env.step_dt)[
            :, self.body_ids
        ]
        last_air_time = self.contact_sensor.data.last_air_time[:, self.body_ids]
        contact = self.last_air_time != last_air_time
        self.last_air_time = last_air_time
        self.reward = torch.sum(
            (last_air_time - self.thres).clamp_max(0.0) * contact, dim=1, keepdim=True
        )
        self.reward *= ~self.env.command_manager.is_standing_env
        return self.reward


class feet_contact_count(Reward):
    def __init__(self, env, body_names: str, weight: float):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)
        self.first_contact = torch.zeros(
            self.num_envs, len(self.body_ids), device=self.env.device
        )

    def compute(self):
        self.first_contact[:] = self.contact_sensor.compute_first_contact(
            self.env.step_dt
        )[:, self.body_ids]
        return self.first_contact.sum(1, keepdim=True)


class root_upright(Reward):
    def __init__(self, env, weight: float):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene["robot"]
        self.update()
    
    def update(self):
        self.projected_gravity_b = self.asset.data.projected_gravity_b

    def compute(self) -> torch.Tensor:
        rew = 1. - self.projected_gravity_b[:, :2].square().sum(1)
        return rew.reshape(self.num_envs, 1)


class base_height_l1(Reward[Twist]):
    def __init__(self, env, weight: float, target_height: Optional[float] = None):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene["robot"]
        self.target_height = target_height

    def compute(self) -> torch.Tensor:
        if self.target_height is None:
            target_height = self.command_manager.cmd_base_height
        else:
            target_height = self.target_height
        root_pos_w = self.asset.data.root_pos_w
        height = root_pos_w[:, 2] - self.env.get_ground_height_at(root_pos_w)
        error_l1 = (height.unsqueeze(1) - target_height).abs()
        return - error_l1.reshape(self.num_envs, 1)


class base_height_exp(Reward[Twist]):
    def __init__(self, env, weight: float, target_height: Optional[float] = None):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene["robot"]
        self.target_height = target_height
    
    def compute(self) -> torch.Tensor:
        if self.target_height is None:
            target_height = self.command_manager.cmd_base_height
        else:
            target_height = self.target_height
        root_pos_w = self.asset.data.root_pos_w
        height = root_pos_w[:, 2] - self.env.get_ground_height_at(root_pos_w)
        error_l2 = (height.unsqueeze(1) - target_height).square()
        rew = torch.exp(-error_l2 / 0.2)
        return rew.reshape(self.num_envs, 1)


class single_foot_contact(Reward):
    def __init__(self, env, body_names: str, margin: float, weight: float):
        super().__init__(env, weight)
        # self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)
        self.margin = margin

    def compute(self) -> torch.Tensor:
        in_contact = self.contact_sensor.data.current_contact_time[:, self.body_ids] > self.margin
        single_contact = torch.where(torch.sum(in_contact, dim=1) == 1, 0., -1.)
        return single_contact.reshape(self.num_envs, 1)


class is_standing_env(Reward):
    def __init__(self, env, weight: float):
        super().__init__(env, weight)

    def compute(self) -> torch.Tensor:
        return self.env.command_manager.is_standing_env.reshape(self.num_envs, 1)


class stance_width(Reward):
    def __init__(self, env, weight: float, target_width=0.15):
        """penalize stance width smaller than target_width"""
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene["robot"]
        self.target_width = target_width

    def compute(self) -> torch.Tensor:
        front_width = (
            self.asset.data.feet_pos_b[:, [0, 1], 0]
            .diff(dim=1)
            .norm(dim=1, keepdim=True)
        )
        back_width = (
            self.asset.data.feet_pos_b[:, [2, 3], 0]
            .diff(dim=1)
            .norm(dim=1, keepdim=True)
        )
        width = torch.cat([front_width, back_width], dim=1)
        return -(self.target_width - width).clamp_min(0.0).sum(1, keepdim=True)


class feet_swing_height(Reward):
    def __init__(self, env, target_height: float, weight: float):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene["robot"]
        self.target_height = target_height
        self.feet_ids = self.asset.find_bodies(".*foot.*")[0]

    def update(self):
        self.feet_pos_b = quat_rotate_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            self.asset.data.body_pos_w[:, self.feet_ids]
            - self.asset.data.root_pos_w.unsqueeze(1),
        )
        self.feet_vel_b = quat_rotate_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            self.asset.data.body_lin_vel_w[:, self.feet_ids],
        )
        if not hasattr(self.asset, "feet_height"):
            self.feet_height = self.asset.data.body_pos_w[:, self.feet_ids, 2]
        else:
            self.feet_height = self.asset.data.feet_height

    def compute(self) -> torch.Tensor:
        hight_error = (self.feet_height - self.target_height).abs()
        lateral_speed = (
            self.feet_vel_b[:, :, :2].square().sum(-1)
            + self.asset.data.body_ang_vel_w[:, self.feet_ids, 2].square()
        )
        return -(hight_error * lateral_speed).sum(1, keepdim=True)


class head_clearance(Reward):
    def __init__(self, env, target_height: float, weight: float):
        super().__init__(env, weight)
        self.target_height = target_height
        self.asset: Articulation = self.env.scene["robot"]
        self.head_height: torch.Tensor = self.asset.data.head_height

    def compute(self) -> torch.Tensor:
        return (self.head_height - self.target_height).clamp_max(0.0)


class com_support(Reward):
    def __init__(self, env, weight: float):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene["robot"]
        self.feet_ids = self.asset.find_bodies(".*foot")[0]

    def compute(self) -> torch.Tensor:
        feet_center = self.asset.data.body_pos_w[:, self.feet_ids].mean(1)
        error = (
            (self.asset.data.root_pos_w[:, :2] - feet_center[:, :2])
            .square()
            .sum(1, keepdim=True)
        )
        return torch.exp(-error / 0.2)


class joint_limits(Reward):
    def __init__(self, env, joint_names: str, offset: float, weight: float):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids = self.asset.find_joints(joint_names)[0]
        self.joint_limits = self.asset.data.joint_limits[:, self.joint_ids].clone()
        self.joint_limits_max = self.joint_limits[:, :, 1] - offset
        self.joint_limits_min = self.joint_limits[:, :, 0] + offset

    def compute(self) -> torch.Tensor:
        joint_pos = self.asset.data.joint_pos[:, self.joint_ids]
        violation_min = (joint_pos - self.joint_limits_min).clamp_max(0.0)
        violation_max = (self.joint_limits_max - joint_pos).clamp_max(0.0)
        return (violation_min + violation_max).sum(1, keepdim=True)


class oscillator(Reward):
    def __init__(
        self,
        env,
        feet_names: str = ".*_foot",
        omega_range=(2., 2.),
        margin: float = 0.0,
        weight=1.0,
    ):
        super().__init__(env, weight)
        self.margin = margin
        self.target_swing_height = 0.08

        self.asset: Articulation = self.env.scene["robot"]
        self.art_feet_ids = self.asset.find_bodies(feet_names)[0]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.command_manager: Command2 = self.env.command_manager

        self.feet_ids, feet_names = self.contact_sensor.find_bodies(feet_names)
        self.mass = self.asset.data.default_mass[0].sum().to(self.device)
        self.gravity = self.mass * 9.81

        # if not hasattr(self.asset, "phi"):
        #     self.asset.phi = torch.zeros(self.num_envs, 4, device=self.device)
        #     self.asset.phi_dot = torch.zeros(self.num_envs, 4, device=self.device)
        # self.asset.phi[:, 0] = torch.pi
        # self.asset.phi[:, 3] = torch.pi
        self.grf_substep = torch.zeros(
            self.num_envs,
            self.env.decimation,
            len(self.feet_ids),
            device=self.device,
        )
        self.omega_range = omega_range
        self.omega = torch.zeros(self.num_envs, 1, device=self.device)
        self.omega.uniform_(*self.omega_range).mul_(torch.pi)

        self.rest_target = torch.pi * 3 / 2
        self.keep_steping = torch.zeros(
            self.num_envs, 1, dtype=bool, device=self.device
        )

    # def reset(self, env_ids):
    #     self.keep_steping[env_ids] = (torch.rand(len(env_ids), 1, device=self.device) < 0.)
    #     self.asset.phi_dot[env_ids] = self.omega[env_ids]

    def post_step(self, substep):
        grf = self.contact_sensor.data.net_forces_w[:, self.feet_ids].norm(dim=-1)
        grf += self.asset._external_force_b[:, self.art_feet_ids].norm(dim=-1)
        self.grf_substep[:, substep] = grf

    def update(self):
        self.grf = self.grf_substep.mean(1) / self.gravity
        # inp = (
        #     (~self.command_manager.is_standing_env)
        #     | self.keep_steping
        # )
        # correction = self.trot(self.asset.phi, self.asset.phi_dot)
        # phi_dot = torch.where(
        #     inp,
        #     self.omega + correction,
        #     self.stand(self.asset.phi, self.asset.phi_dot),
        # )
        
        # self.asset.phi_dot = phi_dot
        # self.asset.phi += self.asset.phi_dot * self.env.step_dt
        # self.asset.phi = torch.where((self.asset.phi > torch.pi * 2).all(1, True), self.asset.phi - torch.pi * 2, self.asset.phi)

    def compute(self):
        phi_sin = self.asset.phi.sin()
        feet_height = self.asset.data.feet_height.clamp_max(self.target_swing_height)
        r = (
            (feet_height - self.grf.clamp_max(0.4))
            * phi_sin
            * (phi_sin.abs() > self.margin)
        )
        return r.sum(1, True)

    def stand(self, phi: torch.Tensor, phi_dot: torch.Tensor,):
        two_pi = torch.pi * 2
        target = self.rest_target
        dt = self.env.step_dt
        a = ((phi % two_pi) < target - 1e-4) & (((phi + phi_dot * dt) % two_pi) > target + 1e-4)
        b = ((phi % two_pi) - target).abs() < 1e-4
        phi_dot = torch.where(a, (((target - phi) % two_pi) / dt), phi_dot)
        return phi_dot * (~b)

    def trot(self, phi: torch.Tensor, phi_dot: torch.Tensor):
        phi_dot = torch.zeros_like(phi)
        phi_dot[:, 0] = (phi[:, 3] - phi[:, 0]) + (phi[:, 1] + torch.pi - phi[:, 0]) 
        phi_dot[:, 1] = (phi[:, 2] - phi[:, 1]) + (phi[:, 0] - torch.pi - phi[:, 1]) 
        phi_dot[:, 2] = (phi[:, 1] - phi[:, 2]) + (phi[:, 0] - torch.pi - phi[:, 2])
        phi_dot[:, 3] = (phi[:, 0] - phi[:, 3]) + (phi[:, 1] + torch.pi - phi[:, 3])
        return phi_dot


class oscillator_biped(Reward):
    def __init__(self, env, weight):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene["robot"]
        self.gravity = self.asset.data.default_mass[0].sum().item() * 9.81
        self.contact_forces: ContactSensor = self.env.scene["contact_forces"]
        self.feet_ids = self.contact_forces.find_bodies(".*_ankle_roll_link")[0]

    def compute(self):
        self.sin_phase = self.asset.phi.sin()
        grf = self.contact_forces.data.net_forces_w[:, self.feet_ids].norm(dim=-1)
        r = (-grf/self.gravity * self.sin_phase).clamp_max(0.8).sum(1, True)
        return r


class quadruped_stand(Reward):
    def __init__(self, env, feet_names: str, weight: float):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene["robot"]
        self.feet_ids = self.asset.find_bodies(feet_names)[0]
        if not hasattr(self.env.command_manager, "is_standing_env"):
            raise ValueError("is_standing_env is not defined in command_manager")
        self.command_manager = self.env.command_manager

    def compute(self):
        jpos_errors = (self.asset.data.joint_pos - self.asset.data.default_joint_pos).abs()
        feet_pos_w = self.asset.data.body_pos_w[:, self.feet_ids]
        feet_pos_b = quat_rotate_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            feet_pos_w - self.asset.data.root_pos_w.unsqueeze(1)
        )
        front_symmetry = feet_pos_b[:, [0, 1], 1].sum(dim=1, keepdim=True).abs()
        back_symmetry = feet_pos_b[:, [2, 3], 1].sum(dim=1, keepdim=True).abs()
        cost = - (jpos_errors.sum(dim=1, keepdim=True) + front_symmetry + back_symmetry)

        return cost * self.command_manager.is_standing_env.reshape(self.num_envs, 1)


class lateral_swing_height(Reward):
    def __init__(self, env, weight: float, feet_names: str, target_height: float):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene["robot"]
        self.feet_ids = self.asset.find_bodies(feet_names)[0]
        self.target_height = target_height
        
    def compute(self):
        feet_pos_w = self.asset.data.body_pos_w[:, self.feet_ids]
        feet_lin_vel_w = self.asset.data.body_lin_vel_w[:, self.feet_ids]
        feet_lin_vel_b = quat_rotate_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            feet_lin_vel_w
        )
        feet_height_w = feet_pos_w[:, :, 2] - self.env.get_ground_height_at(feet_pos_w) # [N, 4]
        rew = torch.where(
            feet_lin_vel_b[:, :, 1].abs() > 0.1,
            1.0 + (feet_height_w/self.target_height - self.target_height).clamp_max(0.0),
            0.
        )
        return rew.sum(1, True) / len(self.feet_ids)


class action_rate_l2(Reward):
    """Penalize the rate of change of the action"""
    def __init__(self, env, weight: float, key: str="action"):
        super().__init__(env, weight)
        self.action_manager = self.env.input_managers[key]
        assert self.action_manager.action_buf.shape[-1] == self.action_manager.action_dim
    
    def compute(self) -> torch.Tensor:
        action_buf = self.action_manager.action_buf
        action_diff = action_buf[:, 0] - action_buf[:, 1]
        rew = - action_diff.square().sum(dim=-1, keepdim=True)
        return rew


class action_rate2_l2(Reward):
    """Penalize the second order rate of change of the action"""
    def __init__(self, env, weight: float, key: str="action"):
        super().__init__(env, weight)
        self.action_manager = self.env.input_managers[key]
        assert self.action_manager.action_buf.shape[-1] == self.action_manager.action_dim
    
    def compute(self) -> torch.Tensor:
        action_buf = self.action_manager.action_buf
        action_diff = (
            action_buf[:, 0] - 2 * action_buf[:, 1] + action_buf[:, 2]
        )
        rew = - action_diff.square().sum(dim=-1, keepdim=True)
        return rew


class support_polygon(Reward):
    def __init__(self, env, margin: float, weight: float):
        super().__init__(env, weight)
        self.margin = margin
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.feet_ids, self.feet_names = self.asset.find_bodies(".*_foot")
        self.feet_ids_contact = self.contact_sensor.find_bodies(".*_foot")[0]

        body_masses = self.asset.data.default_mass.to(self.device)
        mass_total = body_masses.sum(dim=1)
        self.body_mass_ratio = body_masses / mass_total.unsqueeze(-1)
    
    def update(self):
        # self.com_pos_w = (self.asset.data.body_com_pos_w * self.body_mass_ratio.unsqueeze(-1)).sum(dim=1)
        self.com_pos_w = self.asset.data.root_com_pos_w
        self.feet_pos_w = self.asset.data.body_pos_w[:, self.feet_ids]
    
    def compute(self):
        feet_pos_b = quat_rotate_inverse(
            self.asset.data.root_quat_w.unsqueeze(1),
            self.feet_pos_w - self.com_pos_w.unsqueeze(1)
        )        
        in_contact = self.contact_sensor.data.net_forces_w[:, self.feet_ids_contact].norm(dim=-1, keepdim=True) > 0.1
        rew = ((feet_pos_b.abs() - self.margin).clamp_max(0.) * in_contact).sum(-1)
        rew = rew.min(dim=1).values
        return rew.reshape(self.num_envs, 1)
    
    def debug_draw(self):
        if self.env.backend == "isaac":
            self.env.debug_draw.vector(
                self.com_pos_w,
                torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, 3),
                color=(1.0, 0.0, 0.0, 1.0),
            )


def normalize(x: torch.Tensor):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def shaped_error(error: torch.Tensor):
    """
    Shaped error for reward shaping.
    """
    return torch.maximum(error.abs(), error.square())
