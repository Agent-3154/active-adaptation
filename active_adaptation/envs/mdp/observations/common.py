import torch
import numpy as np
import einops
from typing import Tuple, TYPE_CHECKING
from typing_extensions import override

import active_adaptation
from .base import ObservationV2
from active_adaptation.utils.math import (
    quat_rotate,
    quat_rotate_inverse,
    yaw_quat,
    random_noise,
)
import active_adaptation.utils.symmetry as sym_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from active_adaptation.envs.env_base import _EnvBase


class root_pose_w(ObservationV2):
    """Root link pose (position, quaternion) in world frame."""

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]

    @override
    def compute(self) -> torch.Tensor:
        return self.asset.data.root_link_pose_w.reshape(self.num_envs, -1)


class root_state_w(ObservationV2):
    """Root link state (position, quaternion, linear velocity, angular velocity) in world frame."""

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]

    @override
    def compute(self) -> torch.Tensor:
        return self.asset.data.root_link_state_w.reshape(self.num_envs, -1)


class root_linacc_substep(ObservationV2):
    def __init__(self, steps: int | None = None, flatten: bool = False):
        self.steps = steps
        self.flatten = flatten

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        steps = self.steps if self.steps is not None else self.env.decimation
        shape = (self.num_envs, steps, 3)
        self.lin_acc_substep = torch.zeros(shape, device=self.device)

    @override
    def post_step(self, substep):
        root_link_quat_w = self.asset.data.root_link_quat_w
        self.lin_acc_substep[:, substep] = quat_rotate_inverse(
            root_link_quat_w,
            self.asset.data.body_lin_vel_w[:, 0],
        )

    @override
    def compute(self):
        if self.flatten:
            return self.lin_acc_substep.reshape(self.num_envs, -1)
        return self.lin_acc_substep


class command(ObservationV2):
    def __init__(self, key: str | None = None):
        self.key = key

    @override
    def compute(self):
        if self.key is not None:
            return self.command_manager.command(self.key)
        return self.command_manager.command

    @override
    def symmetry_transform(self):
        if self.key is not None:
            return self.command_manager.symmetry_transform(self.key)
        return self.command_manager.symmetry_transform()


class root_angvel_b(ObservationV2):
    def __init__(self, steps: int = 1, noise_std: float = 0.0, yaw_only: bool = False):
        self.steps = steps
        self.noise_std = noise_std
        self.yaw_only = yaw_only

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.buffer = torch.zeros((self.num_envs, self.steps, 3), device=self.device)
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
        transform = sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[-1.0, 1.0, -1.0])
        return transform.repeat(self.steps)


class root_gyro_substep(ObservationV2):
    def __init__(self, steps: int | None = None, flatten: bool = False):
        self.steps = steps
        self.flatten = flatten

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        steps = self.steps if self.steps is not None else self.env.decimation
        shape = (self.num_envs, steps, 3)
        self.gyro = torch.zeros(shape, device=self.device)

    @override
    def post_step(self, substep):
        self.gyro[:, substep] = self.asset.data.root_link_ang_vel_b

    @override
    def compute(self):
        if self.flatten:
            return self.gyro.reshape(self.num_envs, -1)
        return self.gyro


class root_gyro_multistep(ObservationV2):
    def __init__(self, steps: int = 4, noise_std: float = 0.0):
        self.steps = steps
        self.noise_std = noise_std

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.gyro_multistep = torch.zeros((self.num_envs, self.steps, 3), device=self.device)

    @override
    def update(self):
        self.gyro_multistep = self.gyro_multistep.roll(1, dims=1)
        self.gyro_multistep[:, 0] = random_noise(
            self.asset.data.root_link_ang_vel_b, self.noise_std
        )

    @override
    def compute(self):
        return self.gyro_multistep.reshape(self.num_envs, -1)


class projected_gravity_b(ObservationV2):
    def __init__(self, noise_std: float = 0.0):
        self.noise_std = noise_std

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.init_quat = self.asset.data.root_link_quat_w.clone()

    @override
    def compute(self):
        projected_gravity_b = self.asset.data.projected_gravity_b
        noise = torch.randn_like(projected_gravity_b).clip(-3.0, 3.0) * self.noise_std
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


class gravity_multistep(ObservationV2):
    def __init__(self, steps: int = 4, interval: int = 1, noise_std: float = 0.0):
        self.steps = steps
        self.interval = interval
        self.noise_std = noise_std

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.gravity_multistep = torch.zeros(
            (self.num_envs, self.steps * self.interval, 3), device=self.device
        )

    @override
    def update(self):
        gravity = random_noise(self.asset.data.projected_gravity_b, self.noise_std)
        gravity = gravity / gravity.norm(dim=-1, keepdim=True)
        self.gravity_multistep = self.gravity_multistep.roll(1, dims=1)
        self.gravity_multistep[:, 0] = gravity

    @override
    def compute(self):
        gravity_vector = self.gravity_multistep[:, :: self.interval]
        return gravity_vector.reshape(self.num_envs, -1)

    @override
    def symmetry_transform(self):
        transform = sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1, -1, 1])
        return transform.repeat(self.steps)


class gravity_substep(ObservationV2):
    def __init__(self, steps: int | None = None, flatten: bool = False):
        self.steps = steps
        self.flatten = flatten

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        steps = self.steps if self.steps is not None else self.env.decimation
        shape = (self.num_envs, steps, 3)
        self.gravity = torch.zeros(shape, device=self.device)

    @override
    def post_step(self, substep):
        self.gravity[:, substep] = self.asset.data.projected_gravity_b

    @override
    def compute(self):
        if self.flatten:
            return self.gravity.reshape(self.num_envs, -1)
        return self.gravity


class root_linvel_b(ObservationV2):
    def __init__(self, yaw_only: bool = False):
        self.yaw_only = yaw_only

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
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


class prev_actions(ObservationV2):
    def __init__(self, key: str = "action", steps: int = 1, flatten: bool = True):
        self.key = key
        self.steps = steps
        self.flatten = flatten

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.action_manager = self.env.input_managers[self.key]

    def compute(self):
        action_buf = self.action_manager.action_buf[:, : self.steps]
        if self.flatten:
            return action_buf.reshape(self.num_envs, -1)
        return action_buf

    @override
    def symmetry_transform(self):
        transform = self.action_manager.symmetry_transform()
        return transform.repeat(self.steps)


class cum_error(ObservationV2):
    def compute(self) -> torch.Tensor:
        return self.command_manager._cum_error

    def fliplr(self, obs: torch.Tensor) -> torch.Tensor:
        return obs


class clock(ObservationV2):
    def __init__(self, frequencies: list[int] = [1, 2, 4]):
        self.frequencies = frequencies

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.frequencies = torch.as_tensor(self.frequencies, device=self.device).unsqueeze(0)

    def compute(self) -> torch.Tensor:
        t = self.env.episode_length_buf * self.env.step_dt
        t = t.reshape(self.num_envs, 1) * self.frequencies
        return torch.cat([t.sin(), t.cos()], dim=1)


class command_mode(ObservationV2):
    def compute(self) -> torch.Tensor:
        return self.command_manager.command_mode
