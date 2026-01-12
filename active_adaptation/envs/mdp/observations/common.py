import torch
import numpy as np
import einops
from typing import Tuple, TYPE_CHECKING
from typing_extensions import override

from isaaclab.utils.string import resolve_matching_names
import active_adaptation
from active_adaptation.envs.mdp.base import Observation
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse, yaw_quat, EMA
import active_adaptation.utils.symmetry as sym_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor, Imu

if active_adaptation.get_backend() == "isaac":
    import isaaclab.sim as sim_utils
    from isaaclab.terrains.trimesh.utils import make_plane
    from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
    from pxr import UsdGeom, UsdPhysics


class root_pose_w(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]

    @override
    def compute(self):
        return self.asset.data.root_link_pose_w.reshape(self.num_envs, -1)


class root_linacc_substep(Observation):
    def __init__(self, env, steps: int=None, flatten: bool=False):
        super().__init__(env)
        self.flatten = flatten
        self.asset: Articulation = self.env.scene["robot"]
        if steps is None:
            steps = self.env.decimation
        shape = (self.num_envs, steps, 3)
        self.lin_acc_substep = torch.zeros(shape, device=self.env.device)

    @override
    def post_step(self, substep):
        root_link_quat_w = self.asset.data.root_link_quat_w
        self.lin_acc_substep[:, substep] = quat_rotate_inverse(
            root_link_quat_w,
            self.asset.data.body_lin_vel_w[:, 0]
        )

    @override
    def compute(self):
        if self.flatten:
            return self.lin_acc_substep.reshape(self.num_envs, -1)
        else:
            return self.lin_acc_substep


class command(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.command_manager = self.env.command_manager

    @override
    def compute(self):
        return self.command_manager.command
    
    @override
    def symmetry_transform(self):
        return self.command_manager.symmetry_transform()


class command_hidden(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.command_manager = self.env.command_manager
    
    @override
    def compute(self):
        return self.command_manager.command_hidden
    
    @override
    def symmetry_transform(self):
        transform = sym_utils.SymmetryTransform(
            perm=torch.arange(3), 
            signs=[1, -1, 1]
        )
        return sym_utils.SymmetryTransform.cat([
            transform.repeat(3),
            transform.repeat(3),
            sym_utils.SymmetryTransform(torch.arange(3), torch.tensor([-1, -1, -1])),
            sym_utils.SymmetryTransform(torch.arange(3), torch.tensor([-1, -1, -1])),
        ])


class root_angvel_b(Observation):
    def __init__(self, env, steps: int=1, noise_std: float=0., yaw_only: bool=False):
        super().__init__(env)
        self.steps = steps
        self.asset: Articulation = self.env.scene["robot"]
        self.noise_std = noise_std
        self.yaw_only = yaw_only
        self.buffer = torch.zeros((self.num_envs, steps, 3), device=self.device)
        self.update()
    
    @override
    def reset(self, env_ids):
        self.buffer[env_ids] = 0.0
    
    @override
    def update(self):
        if self.yaw_only:
            self.quat = yaw_quat(self.asset.data.root_link_quat_w)
        else:
            self.quat = self.asset.data.root_link_quat_w
        self.root_angvel_w = self.asset.data.root_com_ang_vel_w.clone()
        ang_vel_w = random_noise(self.root_angvel_w, self.noise_std) 
        ang_vel_b = quat_rotate_inverse(self.quat, ang_vel_w)
        self.buffer = self.buffer.roll(1, dims=1)
        self.buffer[:, 0] = ang_vel_b

    @override
    def compute(self) -> torch.Tensor:
        return self.buffer.reshape(self.num_envs, -1)
    
    @override
    def symmetry_transform(self):
        # left-right symmetry: flip only roll and yaw
        transform = sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[-1., 1., -1.])
        return transform.repeat(self.steps)


class root_gyro_substep(Observation):
    def __init__(self, env, steps: int=None,flatten: bool=False):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        if steps is None:
            steps = self.env.decimation
        shape = (self.num_envs, steps, 3)
        self.gyro = torch.zeros(shape, device=self.device)
        self.flatten = flatten

    @override
    def post_step(self, substep):
        self.gyro[:, substep] = self.asset.data.root_link_ang_vel_b
    
    @override
    def compute(self):
        if self.flatten:
            return self.gyro.reshape(self.num_envs, -1)
        else:
            return self.gyro

class root_gyro_multistep(Observation):
    def __init__(self, env, steps: int=4, noise_std: float=0.):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.noise_std = noise_std
        self.gyro_multistep = torch.zeros((self.num_envs, steps, 3), device=self.device)
    
    @override
    def update(self):
        self.gyro_multistep = self.gyro_multistep.roll(1, dims=1)
        self.gyro_multistep[:, 0] = random_noise(self.asset.data.root_link_ang_vel_b, self.noise_std)
    
    @override
    def compute(self):
        return self.gyro_multistep.reshape(self.num_envs, -1)


class projected_gravity_b(Observation):
    def __init__(self, env, noise_std: float=0.):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.init_quat = self.asset.data.root_link_quat_w.clone()
        self.noise_std = noise_std
    
    @override
    def compute(self):
        # projected_gravity_b = quat_rotate_inverse(self.init_quat, self.asset.data.projected_gravity_b)
        projected_gravity_b = self.asset.data.projected_gravity_b
        noise = torch.randn_like(projected_gravity_b).clip(-3., 3.) * self.noise_std
        projected_gravity_b += noise
        return projected_gravity_b / projected_gravity_b.norm(dim=-1, keepdim=True)

    @override
    def symmetry_transform(self):
        transform = sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1, -1, 1])
        return transform
    
    def lerp(self, obs_tm1, obs_t, t):
        gravity = torch.lerp(obs_tm1, obs_t, t)
        gravity = gravity / gravity.norm(dim=-1, keepdim=True)
        return gravity


class gravity_multistep(Observation):
    def __init__(
        self, env, steps: int=4, interval: int=1, noise_std: float=0.):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.steps = steps
        self.interval = interval
        self.noise_std = noise_std
        self.gravity_multistep = torch.zeros((self.num_envs, self.steps * interval, 3), device=self.device)
    
    @override
    def update(self):
        gravity = random_noise(self.asset.data.projected_gravity_b, self.noise_std)
        gravity = gravity / gravity.norm(dim=-1, keepdim=True)
        self.gravity_multistep = self.gravity_multistep.roll(1, dims=1)
        self.gravity_multistep[:, 0] = gravity
    
    @override
    def compute(self):
        gravity_vector = self.gravity_multistep[:, ::self.interval]
        return gravity_vector.reshape(self.num_envs, -1)
    
    @override
    def symmetry_transform(self):
        transform = sym_utils.SymmetryTransform(
            perm=torch.arange(3),
            signs=[1, -1, 1]
        )
        return transform.repeat(self.steps)


class gravity_substep(Observation):
    def __init__(self, env, steps: int=None, flatten: bool=False):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        if steps is None:
            steps = self.env.decimation
        shape = (self.num_envs, steps, 3)
        self.gravity = torch.zeros(shape, device=self.device)
        self.flatten = flatten
    
    @override
    def post_step(self, substep):
        self.gravity[:, substep] = self.asset.data.projected_gravity_b
    
    @override
    def compute(self):
        if self.flatten:
            return self.gravity.reshape(self.num_envs, -1)
        else:
            return self.gravity


class root_linvel_b(Observation):
    def __init__(self, env, yaw_only: bool=False):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.yaw_only = yaw_only

        self.quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.linvel_w = torch.zeros(self.num_envs, 3, device=self.device)
    
    @override
    def update(self):
        if self.yaw_only:
            self.quat_w = yaw_quat(self.asset.data.root_link_quat_w)
        else:
            self.quat_w = self.asset.data.root_link_quat_w
        self.linvel_w = self.asset.data.root_com_lin_vel_w

    @override
    def compute(self) -> torch.Tensor:
        linvel = quat_rotate_inverse(self.quat_w, self.linvel_w)
        return linvel.reshape(self.num_envs, -1)
    
    @override
    def symmetry_transform(self):
        transform = sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1, -1, 1])
        return transform

    # def debug_draw(self):
    #     if self.env.sim.has_gui() and self.env.backend == "isaac":
    #         if self.body_ids is None:
    #             linvel = self.asset.data.root_link_lin_vel_w
    #         else:
    #             linvel = (self.asset.data.body_lin_vel_w[:, self.body_ids] * self.body_masses).mean(1)
    #         self.env.debug_draw.vector(
    #             self.asset.data.root_pos_w + torch.tensor([0., 0., 0.2], device=self.device),
    #             linvel,
    #             color=(0.8, 0.1, 0.1, 1.)
    #         )


class body_materials(Observation):
    def __init__(self, env, body_names, homogeneous: bool=False):
        super().__init__(env)
        self.homogeneous = homogeneous
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)

        num_shapes_per_body = []
        for link_path in self.asset.root_physx_view.link_paths[0]:
            link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
            num_shapes_per_body.append(link_physx_view.max_shapes)
        cumsum = np.cumsum([0,] + num_shapes_per_body)
        self.shape_ids = torch.cat([
            torch.arange(cumsum[i], cumsum[i+1]) 
            for i in self.body_ids
        ]).to(self.device)

        if self.homogeneous:
            self.shape_ids = self.shape_ids[0]
        
    def compute(self):
        return self.asset.data.body_materials[:, self.shape_ids].reshape(self.num_envs, -1)


class body_mass(Observation):
    def __init__(self, env, body_names, homogeneous: bool=False):
        super().__init__(env)
        self.homogeneous = homogeneous
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
        
        masses = self.asset.root_physx_view.get_masses()[0]
        self.default_mass_total = masses.sum()
        self.masses = torch.zeros_like(masses[self.body_ids], device=self.device)
    
    def startup(self):
        self.masses = (
            self.asset.root_physx_view.get_masses()[:, self.body_ids]
            / self.default_mass_total.sum()
        ).to(self.device)
    
    def compute(self) -> torch.Tensor:
        return self.masses.reshape(self.num_envs, -1)


class prev_actions(Observation):
    def __init__(self, env, key: str="action", steps: int=1, flatten: bool=True):
        super().__init__(env)
        self.steps = steps
        self.flatten = flatten
        self.action_manager = self.env.input_managers[key]
    
    def compute(self):
        action_buf = self.action_manager.action_buf[:, :self.steps]
        if self.flatten:
            return action_buf.reshape(self.num_envs, -1)
        else:
            return action_buf

    def symmetry_transform(self):
        transform = self.action_manager.symmetry_transform()
        return transform.repeat(self.steps)


class incoming_wrench(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.default_mass_total = (
            self.asset.root_physx_view.get_masses()[0]
            .sum().to(self.env.device) * 9.81
        )
        self.child_ids = self.asset.find_bodies(".*_hip")[0]

    def update(self):
        self.forces = self.asset.root_physx_view.get_link_incoming_joint_force()
        self.child_forces = self.forces[:, self.child_ids, :3]
        self.child_forces = quat_rotate(self.asset.data.body_quat_w[:, self.child_ids], self.child_forces)
    
    def compute(self) -> torch.Tensor:
        # measured_forces = self.asset.root_physx_view.get_dof_projected_joint_forces()
        self.forces = self.asset.root_physx_view.get_link_incoming_joint_force()
        return (self.forces / self.default_mass_total).reshape(self.num_envs, -1)

    def debug_draw(self):
        if self.env.sim.has_gui() and self.env.backend == "isaac":
            self.env.debug_draw.vector(
                # self.asset.data.body_link_pos_w[:, self.child_ids],
                self.asset.data.root_pos_w,
                self.child_forces.sum(1),
                color=(0., 0., 1., 1.),
                size=10.
            )



class cum_error(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.command_manager = self.env.command_manager
    
    def compute(self) -> torch.Tensor:
        return self.command_manager._cum_error

    def fliplr(self, obs: torch.Tensor) -> torch.Tensor:
        return obs


class clock(Observation):
    def __init__(self, env, frequencies: list[int]=[1, 2, 4]):
        super().__init__(env)
        self.frequencies = torch.as_tensor(frequencies, device=self.device).unsqueeze(0)
    
    def compute(self) -> torch.Tensor:
        t = (self.env.episode_length_buf * self.env.step_dt)
        t = t.reshape(self.num_envs, 1) * self.frequencies
        return torch.cat([t.sin(), t.cos()], dim=1)

class phase(Observation):
    def __init__(self, env, cycle_range = (1.0, 1.2), deriv: bool=False):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.cycle_range = cycle_range
        self.deriv = deriv
        self.offset_range= [torch.pi/3, 2 * torch.pi/3]
        self.asset.data.phase = torch.zeros(self.num_envs, device=self.device)
        self.phase: torch.Tensor = self.asset.data.phase
        self.omega = torch.zeros(self.num_envs, device=self.device)
        self.offset= torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: torch.Tensor):
        offset = torch.zeros(env_ids.shape, device=self.device)
        offset.uniform_(*self.offset_range)
        offset[torch.rand_like(offset) > 0.5] += torch.pi
        cycle = torch.zeros(env_ids.shape, device=self.device)
        cycle.uniform_(*self.cycle_range)

        self.offset[env_ids] = offset
        self.omega[env_ids] = torch.pi * 2 / cycle

    def update(self):
        self.phase[:] = self.offset + self.env.episode_length_buf * self.omega * self.env.step_dt

    def compute(self) -> torch.Tensor:
        phase_sin = self.phase.sin()
        phase_cos = self.phase.cos()
        if self.deriv:
            return torch.stack([
                phase_sin, self.omega * phase_cos,
                phase_cos, -self.omega * phase_sin
            ], 1)
        else:
            return torch.stack([phase_sin, phase_cos], 1)
    
    def fliplr(self, obs: torch.Tensor) -> torch.Tensor:
        phase_sin = (self.phase + torch.pi).sin()
        phase_cos = (self.phase + torch.pi).cos()
        if self.deriv:
            return torch.stack([
                phase_sin, self.omega * phase_cos,
                phase_cos, -self.omega * phase_sin
            ], 1)
        else:
            return torch.stack([phase_sin, phase_cos], 1)
        
    
class dummy(Observation):
    def __init__(self, env, load_path: str):
        super().__init__(env)
        self.obs: torch.Tensor = torch.load(load_path).to(self.device)
    
    def compute(self) -> torch.Tensor:
        return self.obs.expand(self.num_envs, -1)


def symlog(x: torch.Tensor, a: float=1.):
    return x.sign() * torch.log(x.abs() * a + 1.) / a

def random_noise(x: torch.Tensor, std: float):
    return x + torch.randn_like(x).clamp(-3., 3.) * std

meshes = {}


class root_pos_w(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]

    def compute(self):
        return self.asset.data.root_pos_w


class feet_orientation(Observation):
    def __init__(self, env, feet_names: str):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.feet_id = self.asset.find_bodies(feet_names)[0]
        self.heading_feet = torch.tensor([[[1., 0., 0.]]], device=self.device)
    
    def compute(self):
        self.quat_feet = yaw_quat(self.asset.data.body_quat_w[:, self.feet_id])
        feet_fwd = quat_rotate(self.quat_feet, self.heading_feet)
        return feet_fwd.reshape(self.num_envs, -1)


class oscillator(Observation):
    
    def __init__(self, env, history: bool=False):
        super().__init__(env)
        self.history = history
        self.asset: Articulation = self.env.scene["robot"]
        self.phi_history = torch.zeros(self.num_envs, 4, 4, device=self.device)

    def update(self):
        if self.history:
            self.phi_history = self.phi_history.roll(1, dims=1)
            self.phi_history[:, 0] = self.asset.phi

    def compute(self):
        if self.history:
            phi_sin = self.phi_history.sin().reshape(self.num_envs, -1)
            phi_cos = self.phi_history.cos().reshape(self.num_envs, -1)
        else:
            phi_sin = self.asset.phi.sin()
            phi_cos = self.asset.phi.cos()
        obs = torch.concat([phi_sin, phi_cos, self.asset.phi_dot], dim=-1)
        return obs.reshape(self.num_envs, -1)


class oscillator_biped(Observation):
    def __init__(self, env, omega_range=(2.0, 3.0)):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.asset.phi = torch.zeros(self.num_envs, 2, device=self.device)
        self.omega_range = omega_range
        self.omega = torch.zeros(self.num_envs, 1, device=self.device)

    def reset(self, env_ids: torch.Tensor):
        self.asset.phi[env_ids, 0] = 0.
        self.asset.phi[env_ids, 1] = torch.pi
        omega = torch.zeros(len(env_ids), 1, device=self.device)
        omega.uniform_(self.omega_range[0], self.omega_range[1])# .mul_(torch.pi)
        self.omega[env_ids] = torch.pi * 3 # omega

    def update(self):
        self.asset.phi = (self.asset.phi + self.omega * self.env.step_dt) % (2 * torch.pi)

    def compute(self):
        return torch.cat([self.asset.phi.sin(), self.asset.phi.cos()], dim=-1)


class is_standing(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.command_manager = self.env.command_manager
        if not hasattr(self.command_manager, "is_standing_env"):
            raise RuntimeError(" ")

    def compute(self):
        return self.command_manager.is_standing_env.reshape(self.num_envs, 1)


class feet_contact_multistep(Observation):
    def __init__(self, env, steps: int=4, thres: float=1.):
        super().__init__(env)
        self.thres = thres
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_sensor"]
        self.feet_id = self.asset.find_bodies(".*_foot")[0]
        self.contact = torch.zeros(self.num_envs, steps, device=self.device, dtype=bool)
        self.grf_substep = torch.zeros(self.num_envs, self.env.decimation, device=self.device)
    
    def post_step(self, substep):
        contact_forces = self.contact_sensor.data.net_forces_w[:, self.feet_id]
        self.grf_substep[:, substep] = contact_forces.norm(dim=-1)
    
    def update(self):
        self.contact = self.contact.roll(1, dims=1) 
        self.contact[:, 0] = self.grf_substep.mean(dim=1) > self.thres
    
    def compute(self):
        return self.contact.reshape(self.num_envs, -1)


class cartesian_force(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.feet_ids = torch.as_tensor(self.asset.find_bodies(".*_foot")[0], device=self.device)
        self.feet_names = self.asset.find_bodies(".*_foot")[1]
        # print(self.feet_names)
        self.joint_ids = torch.as_tensor(
            [
                self.asset.find_joints("FL_.*_joint")[0],
                self.asset.find_joints("FR_.*_joint")[0],
                self.asset.find_joints("RL_.*_joint")[0],
                self.asset.find_joints("RR_.*_joint")[0],
            ],
            device=self.device,
        )

    def compute(self):
        self.jacobian = self.asset.root_physx_view.get_jacobians()[:, :, :3, 6:]
        jacobian = einops.rearrange(self.jacobian, "n b c j -> n b j c")
        jacobian = jacobian[:, self.feet_ids.unsqueeze(1), self.joint_ids] # [env, feet, joint, 3]
        torques = self.asset.data.applied_torque[:, self.joint_ids]
        self.force_w = (jacobian.transpose(-1, -2) @ torques.unsqueeze(-1)).squeeze(-1) # [env, feet, 3]
        return self.force_w[:, :, 2].reshape(self.num_envs, -1)

    # def debug_draw(self):
    #     self.feet_pos_w = self.asset.data.body_link_pos_w[:, self.feet_ids]
    #     self.env.debug_draw.vector(
    #         self.feet_pos_w,
    #         - self.force_w / 9.81,
    #         color=(1., 0., 1., 1.)
    #     )


class command_mode(Observation):

    def compute(self) -> torch.Tensor:
        return self.command_manager.command_mode

