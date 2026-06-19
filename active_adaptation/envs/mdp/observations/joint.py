import torch

from typing import TYPE_CHECKING
from typing_extensions import override
from .base import ObservationV2
from active_adaptation.utils.math import normal_noise
from active_adaptation.utils.symmetry import joint_space_symmetry
from active_adaptation.envs.utils import find_joints

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from active_adaptation.envs.env_base import _EnvBase


class joint_observation(ObservationV2):
    def __init__(self, joint_names: str):
        self.joint_names_pattern = joint_names

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.joint_ids, self.joint_names = find_joints(self.asset, self.joint_names_pattern)
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)

    @property
    def num_joints(self):
        return len(self.joint_ids)


class joint_pos(joint_observation):
    def __init__(
        self,
        joint_names: str = ".*",
        noise_std: float = 0.0,
        subtract_offset: bool = False,
    ):
        super().__init__(joint_names)
        self.noise_std = noise_std
        self.subtract_offset = subtract_offset

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids]

    @override
    def compute(self):
        joint_pos = self.asset.data.joint_pos[:, self.joint_ids]
        if self.subtract_offset:
            joint_pos = joint_pos - self.default_joint_pos
        if self.noise_std > 0:
            joint_pos = normal_noise(joint_pos, self.noise_std)
        return joint_pos.reshape(self.num_envs, -1)

    @override
    def symmetry_transform(self):
        transform = joint_space_symmetry(self.asset, self.joint_names)
        return transform


class joint_vel(joint_observation):
    def __init__(
        self,
        joint_names: str = ".*",
        noise_std: float = 0.0,
    ):
        super().__init__(joint_names)
        self.noise_std = noise_std

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.default_joint_vel = self.asset.data.default_joint_vel[:, self.joint_ids]

    @override
    def compute(self):
        joint_vel = self.asset.data.joint_vel[:, self.joint_ids]
        if self.noise_std > 0:
            joint_vel = normal_noise(joint_vel, self.noise_std)
        return joint_vel.reshape(self.num_envs, -1)

    @override
    def symmetry_transform(self):
        transform = joint_space_symmetry(self.asset, self.joint_names)
        return transform


class joint_pos_multistep(joint_observation):
    def __init__(
        self,
        joint_names: str = ".*",
        steps: int = 4,
        interval: int = 1,
        noise_std: float = 0.0,
    ):
        super().__init__(joint_names)
        self.steps = steps
        self.interval = interval
        self.noise_std_max = max(noise_std, 0.0)

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.noise_std = torch.zeros(self.num_envs, self.num_joints, device=self.device)
        self.noise_std.uniform_(0.0, self.noise_std_max)

        shape = (self.num_envs, self.steps * self.interval, self.num_joints)
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
        joint_pos = self.joint_pos_multistep[:, :: self.interval]
        return joint_pos.reshape(self.num_envs, -1)

    @override
    def symmetry_transform(self):
        transform = joint_space_symmetry(self.asset, self.joint_names)
        return transform.repeat(self.steps)


class joint_vel_multistep(joint_observation):
    def __init__(
        self,
        joint_names=".*",
        steps: int = 4,
        interval: int = 1,
        noise_std: float = 0.0,
        from_pos: bool = True,
    ):
        super().__init__(joint_names)
        self.steps = steps
        self.interval = interval
        self.noise_std_max = max(noise_std, 0.0)
        self.from_pos = from_pos

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        if self.from_pos and self.env.decimation == 1:
            raise ValueError("Cannot compute joint velocity from position when decimation is 1")

        shape = (self.num_envs, self.steps * self.interval, self.num_joints)
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
        self.noise_std[env_ids] = (
            torch.rand(len(env_ids), self.num_joints, device=self.device) * self.noise_std_max
        )

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
        joint_vel = self.joint_vel_multistep[:, :: self.interval]
        return joint_vel.reshape(self.num_envs, -1)

    @override
    def symmetry_transform(self):
        transform = joint_space_symmetry(self.asset, self.joint_names)
        return transform.repeat(self.steps)


class joint_pos_substep(joint_observation):
    """Only for debugging."""

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        shape = (self.num_envs, self.env.decimation, self.num_joints)
        self.joint_pos_substep = torch.zeros(shape, device=self.device)

    @override
    def post_step(self, substep):
        self.joint_pos_substep[:, substep] = self.asset.data.joint_pos[:, self.joint_ids]

    @override
    def compute(self):
        return self.joint_pos_substep.reshape(self.num_envs, -1)


class joint_vel_substep(joint_observation):
    """Only for debugging."""

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        shape = (self.num_envs, self.env.decimation, self.num_joints)
        self.joint_vel_substep = torch.zeros(shape, device=self.device)

    @override
    def post_step(self, substep):
        self.joint_vel_substep[:, substep] = self.asset.data.joint_vel[:, self.joint_ids]

    @override
    def compute(self):
        return self.joint_vel_substep.reshape(self.num_envs, -1)


class joint_pos_target(joint_observation):
    def __init__(self, joint_names: str, subtract_offset: bool = False):
        super().__init__(joint_names)
        self.subtract_offset = subtract_offset

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids]

    @override
    def compute(self):
        joint_pos_target = self.asset.data.joint_pos_target[:, self.joint_ids]
        if self.subtract_offset:
            joint_pos_target = joint_pos_target - self.default_joint_pos
        return joint_pos_target.reshape(self.num_envs, -1)
