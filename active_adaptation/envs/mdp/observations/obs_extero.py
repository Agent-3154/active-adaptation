import torch
import numpy as np
import einops
from typing import Tuple, TYPE_CHECKING

import active_adaptation
from active_adaptation.envs.mdp.base import Observation
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse, yaw_quat
from active_adaptation.utils.symmetry import SymmetryTransform

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

if active_adaptation.get_backend() == "isaac":
    import isaaclab.sim as sim_utils
    from isaaclab.terrains.trimesh.utils import make_plane
    from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
    from pxr import UsdGeom, UsdPhysics
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg, sim_utils


MESHES = {}


class height_scan(Observation):
    def __init__(
        self,
        env,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        resolution: Tuple[float, float],
        include_xy: bool=False,
        flatten: bool=False,
        noise_scale = 0.005
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.flatten = flatten
        self.noise_scale = noise_scale
        self.include_xy = include_xy
        
        x = torch.linspace(x_range[0], x_range[1], int((x_range[1] - x_range[0]) / resolution[0])+1)
        y = torch.linspace(y_range[0], y_range[1], int((y_range[1] - y_range[0]) / resolution[1])+1)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        self.pos = torch.stack([xx, yy, torch.zeros_like(xx)], dim=-1).to(self.device)
        self.shape = self.pos.shape[:2]
        
        if self.env.backend == "isaac" and self.env.sim.has_gui():
            self.marker = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path=f"/Visuals/Command/height_scan",
                    markers={
                        "scandot": sim_utils.SphereCfg(
                            radius=0.02,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.0)),
                        ),
                    }
                )
            )
            self.marker.set_visibility(True)

    def compute(self):
        root_pos_w = self.asset.data.root_pos_w.reshape(self.num_envs, 1, 1, 3)
        root_quat = yaw_quat(self.asset.data.root_quat_w).reshape(self.num_envs, 1, 1, 4)
        self.offset = quat_rotate(root_quat, self.pos.unsqueeze(0))
        self.height_map_w = self.env.get_ground_height_at(root_pos_w + self.offset)
        height_map = (root_pos_w[:, :, :, 2] - self.height_map_w).clamp(-1., 1.)
        if self.include_xy:
            xy = einops.rearrange(self.pos[..., :2], "X Y C -> C X Y")
            height_map = torch.cat([
                xy.expand(self.num_envs, *xy.shape),
                height_map.reshape(self.num_envs, 1, *self.shape)
            ], dim=1)
        if self.flatten:
            return height_map.reshape(self.num_envs, -1)
        else:
            return height_map.reshape(self.num_envs, -1, *self.shape)
    
    def debug_draw(self):
        if self.env.backend == "isaac":
            pos = self.asset.data.root_pos_w.reshape(self.num_envs, 1, 1, 3) + self.offset
            pos[:, :, :, 2] = self.height_map_w
            self.marker.visualize(pos.reshape(-1, 3))

    def symmetry_transform(self):
        if self.flatten:
            assert not self.include_xy
            perm = torch.arange(self.shape.numel()).reshape(self.shape).flip((1,)).reshape(-1)
            signs = torch.ones(self.shape.numel())
        else:
            perm = torch.arange(self.shape[1]).flip(0) # (N, C, X, Y), flip Y
            signs = torch.ones(self.shape[1])
        return SymmetryTransform(perm=perm, signs=signs)



class forward_scan(Observation):
    def __init__(
        self,
        env,
        hfov: Tuple[float, float],
        vfov: Tuple[float, float],
        resolution: Tuple[int, int],
        max_range: float = 5.0,
        flatten: bool=False,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.ground_mesh = self.env.ground_mesh
        self.max_range = max_range
        self.flatten = flatten
        
        hangles = torch.linspace(hfov[0], hfov[1], resolution[0])
        vangles = torch.linspace(vfov[0], vfov[1], resolution[1])
        vv, hh = torch.meshgrid(vangles, hangles, indexing="ij")
        directions = torch.stack([
            torch.cos(hh) * torch.cos(vv),
            torch.sin(hh) * torch.cos(vv),
            torch.sin(vv),
        ], dim=-1)
        self.shape = directions.shape[:2]
        self.directions = directions.reshape(-1, 3).to(self.device)
        self.num_rays = self.directions.shape[0]

        if self.env.backend == "isaac" and self.env.sim.has_gui():
            self.marker = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path=f"/Visuals/Command/forward_scan",
                    markers={
                        "scandot": sim_utils.SphereCfg(
                            radius=0.02,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.8)),
                        ),
                    }
                )
            )
            self.marker.set_visibility(True)
    
    def compute(self) -> torch.Tensor:
        directions = quat_rotate(
            self.asset.data.root_quat_w.unsqueeze(1),
            self.directions.expand(self.num_envs, self.num_rays, 3)
        )
        ray_starts = self.asset.data.root_pos_w.unsqueeze(1).expand_as(directions)
        ray_hits = raycast_mesh(
            ray_starts=ray_starts.reshape(-1, 3),
            ray_directions=directions.reshape(-1, 3),
            max_dist=self.max_range,
            mesh=self.ground_mesh,
            return_distance=False,
        )[0].reshape(ray_starts.shape)
        ray_distance = (ray_hits - ray_starts).norm(dim=-1)
        ray_distance = ray_distance.nan_to_num(posinf=self.max_range)
        self.ray_hits = ray_starts + ray_distance.unsqueeze(-1) * directions
        if self.flatten:
            return ray_distance.reshape(self.num_envs, -1)
        else:
            return ray_distance.reshape(self.num_envs, 1, *self.shape)
    
    def symmetry_transform(self):
        if self.flatten:
            perm = torch.arange(self.shape.numel())
            perm = perm.reshape(self.shape).flip(1)
            return SymmetryTransform(
                perm=perm.reshape(-1),
                signs=torch.ones(perm.numel())
            )
        else:
            return SymmetryTransform(
                perm=torch.arange(self.shape[1]).flip(0), # (1, H, W), flip W
                signs=torch.ones(self.shape[1])
            )

    def debug_draw(self):
        if self.env.backend == "isaac":
            pos = self.ray_hits.reshape(-1, 3)
            self.marker.visualize(pos)


# class feet_height_map(Observation):
#     def __init__(
#         self, 
#         env, 
#         feet_names=".*_foot", 
#         nomial_height=0.3,
#         resolution: float=0.1,
#         size=[0.15, 0.15],
#     ):
#         super().__init__(env)
#         self.nominal_height = nomial_height
#         self.asset: Articulation = self.env.scene["robot"]
#         self.body_ids, self.body_names = self.asset.find_bodies(feet_names)
#         self.num_feet = len(self.body_ids)
        
#         self.ray_starts = torch.tensor(
#             [
#                 [0., 0., 10.], 
#                 # [0., 0.1, 10.],
#                 # [0., -0.1, 10.],
#                 # [0.1, 0., 10.],
#                 # [-0.1, 0., 10.],
#                 [0.1, 0.1, 10.],
#                 [0.1, -.1, 10.],
#                 [-.1, -.1, 10.],
#                 [-.1, 0.1, 10.],
#             ],
#             device=self.device
#         )
#         self.num_rays = len(self.ray_starts)

#         shape = (self.num_envs, self.num_feet, self.num_rays)
#         self.ray_hits_w = torch.zeros(*shape, 3, device=self.device)
#         self.feet_height_map = torch.zeros(shape, device=self.device)
#         self.asset.data.feet_height = self.feet_height_map[:, :, 0]
#         self.asset.data.feet_height_map = self.feet_height_map
    
#     def update(self):
#         self.feet_pos_w = self.asset.data.body_pos_w[:, self.body_ids]
#         self.feet_quat_w = self.asset.data.body_quat_w[:, self.body_ids]
#         if self.mesh is not None:
#             shape = (self.num_envs, self.num_feet, self.num_rays, -1)
#             ray_starts_w = quat_apply_yaw(
#                 self.feet_quat_w.unsqueeze(-2).expand(shape),
#                 self.ray_starts.reshape(1, 1, -1, 3).expand(shape),
#             )
#             ray_starts_w += self.feet_pos_w.unsqueeze(-2)
#             self.ray_hits_w[:] = raycast_mesh(
#                 ray_starts_w,
#                 self.ray_directions.expand_as(ray_starts_w).clone(),
#                 max_dist=100.,
#                 mesh=self.mesh,
#             )[0]

#             self.feet_height_map[:] = (self.feet_pos_w.unsqueeze(-2)[..., 2] - self.ray_hits_w[..., 2]).nan_to_num(nan=0., posinf=0., neginf=0.)
#         else:
#             self.feet_height_map[:] = self.feet_pos_w.unsqueeze(-2)[..., 2]

#     def compute(self):
#         return self.feet_height_map.reshape(self.num_envs, -1) / self.nominal_height
