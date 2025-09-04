import torch

from typing import TYPE_CHECKING
from active_adaptation.envs.mdp.base import Observation
from active_adaptation.utils.math import normal_noise
from active_adaptation.utils.symmetry import joint_space_symmetry


if TYPE_CHECKING:
    from isaaclab.assets import Articulation


class joint_pos(Observation):
    def __init__(self, env, joint_names: str=".*", noise_std: float=0., subtract_offset: bool=False):
        super().__init__(env)
        self.noise_std = noise_std
        self.subtract_offset = subtract_offset
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.num_joints = len(self.joint_ids)
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids]
        
    def compute(self):
        joint_pos = self.asset.data.joint_pos[:, self.joint_ids]
        if self.subtract_offset:
            joint_pos = joint_pos - self.default_joint_pos
        if self.noise_std > 0:
            joint_pos = normal_noise(joint_pos, self.noise_std)
        return joint_pos.reshape(self.num_envs, -1)
    
    def symmetry_transforms(self):
        transform = joint_space_symmetry(self.asset, self.joint_names)
        return transform


class joint_vel(Observation):
    def __init__(self, env, joint_names: str=".*", noise_std: float=0.):
        super().__init__(env)
        self.noise_std = noise_std
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.num_joints = len(self.joint_ids)
        
    def compute(self):
        joint_vel = self.asset.data.joint_vel[:, self.joint_ids]
        if self.noise_std > 0:
            joint_vel = normal_noise(joint_vel, self.noise_std)
        return joint_vel.reshape(self.num_envs, -1)
    
    def symmetry_transforms(self):
        transform = joint_space_symmetry(self.asset, self.joint_names)
        return transform


class joint_pos_multistep(Observation):
    def __init__(
        self,
        env,
        joint_names: str=".*",
        steps: int=4, 
        interval: int=1,
        noise_std: float=0.,
    ):
        super().__init__(env)
        self.steps = steps
        self.interval = interval
        self.noise_std = max(noise_std, 0.)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.num_joints = len(self.joint_ids)

        shape = (self.num_envs, steps, self.num_joints)
        self.joint_pos_multistep = torch.zeros(shape, device=self.device)
        self.joint_pos_substep = torch.zeros(self.num_envs, 2, self.num_joints, device=self.device)
    
    def post_step(self, substep):
        self.joint_pos_substep[:, substep % 2] = self.asset.data.joint_pos[:, self.joint_ids]
    
    def update(self):
        next_joint_pos_multistep = self.joint_pos_multistep.roll(1, 1)
        next_joint_pos = self.joint_pos_substep.mean(1)
        next_joint_pos_multistep[:, 0] = next_joint_pos
        self.joint_pos_multistep = torch.where(
            (self.env.episode_length_buf % self.interval == 0).reshape(self.num_envs, 1, 1),
            next_joint_pos_multistep,
            self.joint_pos_multistep
        )
    
    def compute(self):
        joint_pos = self.joint_pos_multistep.clone()
        if self.noise_std > 0:
            joint_pos = random_noise(joint_pos, self.noise_std)
        return joint_pos.reshape(self.num_envs, -1)
    
    def symmetry_transforms(self):
        transform = joint_space_symmetry(self.asset, self.joint_names)
        return transform.repeat(self.steps)


class joint_vel_multistep(Observation):
    def __init__(
        self,
        env,
        joint_names=".*",
        steps: int=4,
        noise_std: float=0.,
    ):
        super().__init__(env)
        self.steps = steps
        self.noise_std_max = max(noise_std, 0.)
        self.from_pos = True
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.num_joints = len(self.joint_ids)

        shape = (self.num_envs, steps, self.num_joints)
        
        self.joint_vel_multistep = torch.zeros(shape, device=self.device)
        
        self.noise_std = torch.zeros(self.num_envs, self.num_joints, device=self.device)
        if self.from_pos:
            shape = (self.num_envs, self.env.decimation, self.num_joints)
            self.joint_pos_substep = torch.zeros(shape, device=self.device)
        else:
            shape = (self.num_envs, 2, self.num_joints)
            self.joint_vel_substep = torch.zeros(shape, device=self.device)
    
    def reset(self, env_ids: torch.Tensor):
        self.noise_std[env_ids] = torch.rand(len(env_ids), self.num_joints, device=self.device) * self.noise_std_max

    def post_step(self, substep):
        if self.from_pos:
            self.joint_pos_substep[:, substep] = self.asset.data.joint_pos[:, self.joint_ids]
        else:
            self.joint_vel_substep[:, substep % 2] = self.asset.data.joint_vel[:, self.joint_ids]
    
    def update(self):
        self.joint_vel_multistep = self.joint_vel_multistep.roll(1, 1)
        if self.from_pos:
            joint_vel = self.joint_pos_substep.diff(dim=1).mean(dim=1) / self.env.physics_dt
        else:
            joint_vel = self.joint_vel_substep.mean(dim=1)
        self.joint_vel_multistep[:, 0] = joint_vel
    
    def compute(self):
        joint_vel = self.joint_vel_multistep.clone()
        joint_vel = random_noise(joint_vel, self.noise_std.unsqueeze(1))
        return joint_vel.reshape(self.num_envs, -1)

    def symmetry_transforms(self):
        transform = joint_space_symmetry(self.asset, self.joint_names)
        return transform.repeat(self.steps)


def random_noise(x: torch.Tensor, std: float):
    return x + torch.randn_like(x).clamp(-3., 3.) * std


