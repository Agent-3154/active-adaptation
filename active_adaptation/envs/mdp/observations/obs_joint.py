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

