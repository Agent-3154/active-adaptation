import torch
from typing import TYPE_CHECKING
from typing_extensions import override
from .base import ObservationV2
from active_adaptation.utils.symmetry import cartesian_space_symmetry
from active_adaptation.envs.utils import find_bodies
from active_adaptation.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from active_adaptation.envs.env_base import _EnvBase


class body_observation(ObservationV2):
    def __init__(self, body_names: str):
        self.body_names_pattern = body_names

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.body_ids, self.body_names = find_bodies(self.asset, self.body_names_pattern)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)

    @property
    def num_bodies(self):
        return len(self.body_ids)


class body_height(body_observation):
    @override
    def compute(self):
        body_link_pos_w = self.asset.data.body_link_pos_w[:, self.body_ids]
        ground_height = self.env.get_ground_height_at(body_link_pos_w)
        body_height = body_link_pos_w[:, :, 2] - ground_height
        return body_height.reshape(self.num_envs, -1)

    @override
    def symmetry_transform(self):
        return cartesian_space_symmetry(self.asset, self.body_names, sign=(1,))


class body_link_pos_w(body_observation):
    @override
    def compute(self):
        body_link_pos_w = self.asset.data.body_link_pos_w[:, self.body_ids]
        return body_link_pos_w.reshape(self.num_envs, -1)


class body_link_pos_b(body_observation):
    def __init__(self, body_names: str, yaw_only: bool = False):
        super().__init__(body_names)
        self.yaw_only = yaw_only

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.root_link_pos_w = self.asset.data.root_link_pos_w.unsqueeze(1)
        self.root_link_quat_w = self.asset.data.root_link_quat_w.unsqueeze(1)
        self.body_link_pos_w = self.asset.data.body_link_pos_w[:, self.body_ids]

    @override
    def update(self):
        if self.yaw_only:
            self.root_link_quat_w = yaw_quat(self.asset.data.root_link_quat_w).unsqueeze(1)
        else:
            self.root_link_quat_w = self.asset.data.root_link_quat_w.unsqueeze(1)
        self.root_link_pos_w = self.asset.data.root_link_pos_w.unsqueeze(1)
        self.body_link_pos_w = self.asset.data.body_link_pos_w[:, self.body_ids]

    @override
    def compute(self):
        body_pos_b = quat_rotate_inverse(
            self.root_link_quat_w,
            self.body_link_pos_w - self.root_link_pos_w,
        )
        return body_pos_b.reshape(self.num_envs, -1)

    @override
    def symmetry_transform(self):
        return cartesian_space_symmetry(self.asset, self.body_names)


class body_pos_b(body_link_pos_b):
    """Alias kept for existing task configs that reference ``body_pos_b``."""


class body_vel_b(body_observation):
    def __init__(self, body_names: str, yaw_only: bool = False):
        super().__init__(body_names)
        self.yaw_only = yaw_only

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.root_link_quat_w = self.asset.data.root_link_quat_w.unsqueeze(1)
        self.body_link_vel_w = self.asset.data.body_link_vel_w[:, self.body_ids]

    @override
    def update(self):
        if self.yaw_only:
            self.root_link_quat_w = yaw_quat(self.asset.data.root_link_quat_w).unsqueeze(1)
        else:
            self.root_link_quat_w = self.asset.data.root_link_quat_w.unsqueeze(1)
        self.body_link_vel_w = self.asset.data.body_link_vel_w[:, self.body_ids]

    @override
    def compute(self):
        body_lin_vel_b = quat_rotate_inverse(self.root_link_quat_w, self.body_link_vel_w[:, :, :3])
        body_ang_vel_b = quat_rotate_inverse(self.root_link_quat_w, self.body_link_vel_w[:, :, 3:])
        return torch.cat([body_lin_vel_b, body_ang_vel_b], dim=-1).reshape(self.num_envs, -1)

    @override
    def symmetry_transform(self):
        return cartesian_space_symmetry(self.asset, self.body_names).repeat(2)
