import torch

from typing import TYPE_CHECKING, Literal
from typing_extensions import override
from active_adaptation.envs.mdp.base import Observation
from active_adaptation.utils.math import normal_noise
from active_adaptation.utils.symmetry import joint_space_symmetry
from active_adaptation.assets import get_output_joint_indexing


if TYPE_CHECKING:
    from isaaclab.assets import Articulation


class joint_observation(Observation):
    def __init__(self, env, joint_names: str, output_order: Literal["isaac", "mujoco", "mjlab"] = "isaac"):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.joint_ids = torch.as_tensor(self.joint_ids, device=self.device)
        self.output_indexing, self.output_joint_names = get_output_joint_indexing(
            output_order,
            self.asset.cfg,
            self.joint_names,
            self.device,
        )
    
    @property
    def num_joints(self):
        return len(self.joint_ids)


class joint_pos(joint_observation):
    def __init__(
        self,
        env,
        joint_names: str=".*",
        noise_std: float=0.,
        subtract_offset: bool=False,
        output_order: Literal["isaac", "mujoco", "mjlab"] = "isaac",
    ):
        super().__init__(env, joint_names, output_order)
        self.noise_std = noise_std
        self.subtract_offset = subtract_offset
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids]
        
    @override
    def compute(self):
        joint_pos = self.asset.data.joint_pos[:, self.joint_ids]
        if self.subtract_offset:
            joint_pos = joint_pos - self.default_joint_pos
        if self.noise_std > 0:
            joint_pos = normal_noise(joint_pos, self.noise_std)
        return joint_pos[:, self.output_indexing].reshape(self.num_envs, -1)
    
    @override
    def symmetry_transform(self):
        transform = joint_space_symmetry(self.asset, self.output_joint_names)
        return transform


class joint_vel(joint_observation):
    def __init__(
        self,
        env,
        joint_names: str=".*",
        noise_std: float=0.,
        output_order: Literal["isaac", "mujoco", "mjlab"] = "isaac",
    ):
        super().__init__(env, joint_names, output_order)
        self.noise_std = noise_std
        self.default_joint_vel = self.asset.data.default_joint_vel[:, self.joint_ids]
        
    @override
    def compute(self):
        joint_vel = self.asset.data.joint_vel[:, self.joint_ids]
        if self.noise_std > 0:
            joint_vel = normal_noise(joint_vel, self.noise_std)
        return joint_vel[:, self.output_indexing].reshape(self.num_envs, -1)
    
    @override
    def symmetry_transform(self):
        transform = joint_space_symmetry(self.asset, self.output_joint_names)
        return transform


class joint_pos_multistep(joint_observation):
    def __init__(
        self,
        env,
        joint_names: str=".*",
        steps: int=4, 
        interval: int=1,
        noise_std: float=0.,
        output_order: Literal["isaac", "mujoco", "mjlab"] = "isaac",
    ):
        super().__init__(env, joint_names, output_order)
        self.steps = steps
        self.interval = interval
        self.noise_std_max = max(noise_std, 0.)
        self.noise_std = torch.zeros(self.num_envs, self.num_joints, device=self.device)
        self.noise_std.uniform_(0., self.noise_std_max)

        shape = (self.num_envs, steps * interval, self.num_joints)
        self.joint_pos_multistep = torch.zeros(shape, device=self.device)
        self.joint_pos_substep = torch.zeros(self.num_envs, 2, self.num_joints, device=self.device)

    @override
    def post_step(self, substep):
        self.joint_pos_substep[:, substep % 2] = self.asset.data.joint_pos[:, self.joint_ids]
    
    @override
    def update(self):
        next_joint_pos_multistep = self.joint_pos_multistep.roll(1, 1)
        next_joint_pos = self.joint_pos_substep.mean(1)
        next_joint_pos_multistep[:, 0] = normal_noise(next_joint_pos, self.noise_std)
        self.joint_pos_multistep = next_joint_pos_multistep
    
    @override
    def compute(self):
        joint_pos = self.joint_pos_multistep[:, ::self.interval, self.output_indexing] # [num_envs, steps, joints]
        return joint_pos.reshape(self.num_envs, -1)
    
    @override
    def symmetry_transform(self):
        transform = joint_space_symmetry(self.asset, self.output_joint_names)
        return transform.repeat(self.steps)


class joint_vel_multistep(joint_observation):
    def __init__(
        self,
        env,
        joint_names=".*",
        steps: int=4,
        interval: int=1,
        noise_std: float=0.,
        output_order: Literal["isaac", "mujoco", "mjlab"] = "isaac",
    ):
        super().__init__(env, joint_names, output_order)
        self.steps = steps
        self.interval = interval
        self.noise_std_max = max(noise_std, 0.)
        self.from_pos = True
        shape = (self.num_envs, steps * interval, self.num_joints)
        
        self.joint_vel_multistep = torch.zeros(shape, device=self.device)
        
        self.noise_std = torch.zeros(self.num_envs, self.num_joints, device=self.device)
        if self.from_pos:
            shape = (self.num_envs, self.env.decimation, self.num_joints)
            self.joint_pos_substep = torch.zeros(shape, device=self.device)
        else:
            shape = (self.num_envs, 2, self.num_joints)
            self.joint_vel_substep = torch.zeros(shape, device=self.device)
    
    @override
    def reset(self, env_ids: torch.Tensor):
        self.noise_std[env_ids] = torch.rand(len(env_ids), self.num_joints, device=self.device) * self.noise_std_max

    @override
    def post_step(self, substep):
        if self.from_pos:
            self.joint_pos_substep[:, substep] = self.asset.data.joint_pos[:, self.joint_ids]
        else:
            self.joint_vel_substep[:, substep % 2] = self.asset.data.joint_vel[:, self.joint_ids]
    
    @override
    def update(self):
        self.joint_vel_multistep = self.joint_vel_multistep.roll(1, 1)
        if self.from_pos:
            joint_vel = self.joint_pos_substep.diff(dim=1).mean(dim=1) / self.env.physics_dt
        else:
            joint_vel = self.joint_vel_substep.mean(dim=1)
        self.joint_vel_multistep[:, 0] = normal_noise(joint_vel, self.noise_std)
    
    @override
    def compute(self):
        joint_vel = self.joint_vel_multistep[:, ::self.interval, self.output_indexing]
        return joint_vel.reshape(self.num_envs, -1)

    @override
    def symmetry_transform(self):
        transform = joint_space_symmetry(self.asset, self.output_joint_names)
        return transform.repeat(self.steps)


class joint_pos_substep(joint_observation):
    """Only for debugging"""
    def __init__(self, env, joint_names: str, output_order: Literal["isaac", "mujoco", "mjlab"] = "isaac"):
        super().__init__(env, joint_names, output_order)
        shape = (self.num_envs, self.env.decimation, self.num_joints)
        self.joint_pos_substep = torch.zeros(shape, device=self.device)
    
    @override
    def post_step(self, substep):
        self.joint_pos_substep[:, substep] = self.asset.data.joint_pos[:, self.joint_ids]
    
    @override
    def compute(self):
        return self.joint_pos_substep[:, :, self.output_indexing].reshape(self.num_envs, -1)


class joint_vel_substep(joint_observation):
    """Only for debugging"""
    def __init__(self, env, joint_names: str, output_order: Literal["isaac", "mujoco", "mjlab"] = "isaac"):
        super().__init__(env, joint_names, output_order)
        super().__init__(env)
        shape = (self.num_envs, self.env.decimation, self.num_joints)
        self.joint_vel_substep = torch.zeros(shape, device=self.device)

    @override
    def post_step(self, substep):
        self.joint_vel_substep[:, substep] = self.asset.data.joint_vel[:, self.joint_ids]
    
    @override
    def compute(self):
        return self.joint_vel_substep[:, :, self.output_indexing].reshape(self.num_envs, -1)


class joint_pos_target(joint_observation):
    def __init__(self, env, joint_names: str, subtract_offset: bool=False, output_order: Literal["isaac", "mujoco", "mjlab"] = "isaac"):
        super().__init__(env, joint_names, output_order)
        self.subtract_offset = subtract_offset
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids]

    @override
    def compute(self):
        joint_pos_target = self.asset.data.joint_pos_target
        if self.subtract_offset:
            joint_pos_target = joint_pos_target - self.default_joint_pos
        return joint_pos_target[:, self.output_indexing].reshape(self.num_envs, -1)


# class applied_torque(joint_observation):
    
#     supported_backends = ("isaac",)

#     def __init__(self, env, joint_names: str=".*", output_order: Literal["isaac", "mujoco", "mjlab"] = "isaac"):
#         super().__init__(env, joint_names, output_order=output_order)
#         self.asset: Articulation = self.env.scene["robot"]
#         self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
#         self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
    
#     @override
#     def compute(self) -> torch.Tensor:
#         applied_efforts = self.asset.data.applied_torque
#         return applied_efforts[:, self.joint_ids]
    
#     @override
#     def symmetry_transform(self):
#         transform = joint_space_symmetry(self.asset, self.joint_names)
#         return transform

