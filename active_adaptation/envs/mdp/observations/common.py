import torch
import numpy as np
import einops
from typing import Tuple, TYPE_CHECKING
from typing_extensions import override

import active_adaptation
from .base import Observation
from active_adaptation.utils.math import (
    quat_rotate, quat_rotate_inverse, yaw_quat, random_noise,
)
import active_adaptation.utils.symmetry as sym_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

class root_pose_w(Observation):
    """Root link pose (position, quaternion) in world frame."""
    def __init__(self, env):
        super().__init__(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]

    @override
    def compute(self) -> torch.Tensor:
        return self.asset.data.root_link_pose_w.reshape(self.num_envs, -1)


class root_state_w(Observation):
    """Root link state (position, quaternion, linear velocity, angular velocity) in world frame."""
    def __init__(self, env):
        super().__init__(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]

    @override
    def compute(self) -> torch.Tensor:
        return self.asset.data.root_link_state_w.reshape(self.num_envs, -1)


class root_linacc_substep(Observation):
    def __init__(self, env, steps: int=None, flatten: bool=False):
        super().__init__(env)
        self.flatten = flatten
        self.asset: Articulation = self.env.scene.articulations["robot"]
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


# class command_hidden(Observation):
#     def __init__(self, env):
#         super().__init__(env)
#         self.command_manager = self.env.command_manager
    
#     @override
#     def compute(self):
#         return self.command_manager.command_hidden
    
#     @override
#     def symmetry_transform(self):
#         transform = sym_utils.SymmetryTransform(
#             perm=torch.arange(3), 
#             signs=[1, -1, 1]
#         )
#         return sym_utils.SymmetryTransform.cat([
#             transform.repeat(3),
#             transform.repeat(3),
#             sym_utils.SymmetryTransform(torch.arange(3), torch.tensor([-1, -1, -1])),
#             sym_utils.SymmetryTransform(torch.arange(3), torch.tensor([-1, -1, -1])),
#         ])


class root_angvel_b(Observation):
    def __init__(self, env, steps: int=1, noise_std: float=0., yaw_only: bool=False):
        super().__init__(env)
        self.steps = steps
        self.asset: Articulation = self.env.scene.articulations["robot"]
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
        self.asset: Articulation = self.env.scene.articulations["robot"]
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
        self.asset: Articulation = self.env.scene.articulations["robot"]
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
        self.asset: Articulation = self.env.scene.articulations["robot"]
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
        self.asset: Articulation = self.env.scene.articulations["robot"]
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
        self.asset: Articulation = self.env.scene.articulations["robot"]
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
        self.asset: Articulation = self.env.scene.articulations["robot"]
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

    @override
    def symmetry_transform(self):
        transform = self.action_manager.symmetry_transform()
        return transform.repeat(self.steps)



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



class command_mode(Observation):

    def compute(self) -> torch.Tensor:
        return self.command_manager.command_mode
