import torch
from typing import TYPE_CHECKING, Literal
from typing_extensions import override
from .base import Observation
from active_adaptation.utils.symmetry import cartesian_space_symmetry
from active_adaptation.assets import get_output_body_indexing

if TYPE_CHECKING:
    from isaaclab.assets import Articulation



class body_observation(Observation):
    def __init__(self, env, body_names: str, output_order: Literal["isaac", "mujoco", "mjlab"] = "isaac"):
        super().__init__(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
        self.body_ids = torch.as_tensor(self.body_ids, device=self.device)
        self.output_indexing, self.output_body_names = get_output_body_indexing(
            output_order,
            self.asset.cfg,
            self.body_names,
            self.device,
        )
    
    @property
    def num_bodies(self):
        return len(self.body_ids)


class body_height(body_observation):
    
    @override
    def compute(self):
        body_link_pos_w = self.asset.data.body_link_pos_w[:, self.body_ids]
        ground_height = self.env.get_ground_height_at(body_link_pos_w) # [env, nbody]
        body_height = body_link_pos_w[:, :, 2] - ground_height
        return body_height[:, self.output_indexing].reshape(self.num_envs, -1)

    @override
    def symmetry_transform(self):
        return cartesian_space_symmetry(self.asset, self.output_body_names, sign=(1,))


class body_link_pos_w(body_observation):
    
    @override
    def compute(self):
        body_link_pos_w = self.asset.data.body_link_pos_w[:, self.body_ids]
        return body_link_pos_w[:, self.output_indexing].reshape(self.num_envs, -1)


# class body_pos_b(body_observation):
#     def __init__(self, env, body_names: str, yaw_only: bool=False, output_order: Literal["isaac", "mujoco", "mjlab"] = "isaac"):
#         super().__init__(env, body_names, output_order)
#         self.yaw_only = yaw_only
#         self.root_link_pos_w = self.asset.data.root_link_pos_w.unsqueeze(1)
#         self.root_link_quat_w = self.asset.data.root_link_quat_w.unsqueeze(1)
#         self.body_link_pos_w = self.asset.data.body_link_pos_w[:, self.body_ids]

#     @override
#     def update(self):
#         if self.yaw_only:
#             self.root_link_quat_w = yaw_quat(self.asset.data.root_link_quat_w).unsqueeze(1)
#         else:
#             self.root_link_quat_w = self.asset.data.root_link_quat_w.unsqueeze(1)
#         self.root_link_pos_w = self.asset.data.root_link_pos_w.unsqueeze(1)
#         self.body_link_pos_w = self.asset.data.body_link_pos_w[:, self.body_ids]
        
#     @override
#     def compute(self):
#         body_pos_b = quat_rotate_inverse(
#             self.root_link_quat_w,
#             self.body_link_pos_w - self.root_link_pos_w
#         )
#         return body_pos_b[:, self.output_indexing].reshape(self.num_envs, -1)
    
#     @override
#     def symmetry_transform(self):
#         return cartesian_space_symmetry(self.asset, self.output_body_names)
    

# class body_vel_b(body_observation):

#     def __init__(self, env, body_names: str, yaw_only: bool=False, output_order: Literal["isaac", "mujoco", "mjlab"] = "isaac"):
#         super().__init__(env, body_names, output_order)
#         self.yaw_only = yaw_only
#         self.root_link_quat_w = self.asset.data.root_link_quat_w.unsqueeze(1)
#         self.body_link_vel_w = self.asset.data.body_link_vel_w[:, self.body_ids]
    
#     @override
#     def update(self):
#         if self.yaw_only:
#             self.root_link_quat_w = yaw_quat(self.asset.data.root_link_quat_w).unsqueeze(1)
#         else:
#             self.root_link_quat_w = self.asset.data.root_link_quat_w.unsqueeze(1)
#         self.body_link_vel_w = self.asset.data.body_link_vel_w[:, self.body_ids]
        
#     @override
#     def compute(self):
#         body_lin_vel_b = quat_rotate_inverse(self.root_link_quat_w, self.body_link_vel_w[:, :, :3])
#         body_ang_vel_b = quat_rotate_inverse(self.root_link_quat_w, self.body_link_vel_w[:, :, 3:])
#         return body_lin_vel_b[:, self.output_indexing].reshape(self.num_envs, -1)
    
#     @override
#     def symmetry_transform(self):
#         return cartesian_space_symmetry(self.asset, self.output_body_names)
