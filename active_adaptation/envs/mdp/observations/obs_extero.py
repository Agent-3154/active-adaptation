import torch
import numpy as np
import einops
from typing import Tuple, TYPE_CHECKING, Optional

import active_adaptation
from jaxtyping import Float
from active_adaptation.envs.mdp.base import Observation
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse, yaw_quat
from active_adaptation.utils.symmetry import SymmetryTransform, cartesian_space_symmetry

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

if active_adaptation.get_backend() == "isaac":
    import isaaclab.sim as sim_utils
    from isaaclab.terrains.trimesh.utils import make_plane
    from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
    from pxr import UsdGeom, UsdPhysics
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg, sim_utils


MESHES = {}


def raymap(width: int, height: int, fov: float) -> Float[torch.Tensor, "height width 3"]:
    """
    Generate a raymap for a given width, height, and field of view.
    
    The raymap represents normalized ray directions for a perspective camera model.
    Each pixel corresponds to a ray direction pointing from the camera center through
    that pixel. The rays are in camera space, where +Z is forward, +X is right, and +Y is up.
    
    Args:
        width: The width of the raymap in pixels.
        height: The height of the raymap in pixels.
        fov: The horizontal field of view in radians.

    Returns:
        A tensor of shape (height, width, 3) where the last dimension contains the
        normalized ray direction vector (x, y, z) for each pixel.
    """
    # Create pixel coordinates (u, v) where u ranges from 0 to width-1, v ranges from 0 to height-1
    u = torch.arange(width, dtype=torch.float32)
    v = torch.arange(height, dtype=torch.float32)
    
    # Create meshgrid of pixel coordinates
    uu, vv = torch.meshgrid(u, v, indexing="xy")
    
    # Convert to normalized device coordinates (NDC)
    # x: [-1, 1] from left to right, y: [1, -1] from top to bottom (image coordinates)
    u_ndc = (uu + 0.5) / width * 2.0 - 1.0
    v_ndc = 1.0 - (vv + 0.5) / height * 2.0
    
    # Compute aspect ratio
    aspect_ratio = width / height
    
    # Scale by FOV: horizontal FOV determines the x range, vertical FOV is computed from aspect ratio
    tan_fov_half = torch.tan(torch.tensor(fov / 2.0))
    u_camera = u_ndc * tan_fov_half
    v_camera = v_ndc * tan_fov_half / aspect_ratio
    
    # Create ray directions: (1, x, y) pointing forward in camera space
    x_camera = torch.ones_like(u_camera)
    directions = torch.stack([x_camera, v_camera, u_camera], dim=-1)
    
    # Normalize the directions
    directions = directions / directions.norm(dim=-1, keepdim=True)
    
    return directions


class external_forces(Observation):
    supported_backends = ("isaac",)
    def __init__(self, env, body_names, divide_by_mass: bool=True, scale: float = 1.0):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.forces_b = torch.zeros(self.num_envs, len(self.body_ids) * 3, device=self.device)
        default_mass_total = self.asset.data.default_mass[0].sum() * 9.81
        self.denom = default_mass_total if divide_by_mass else torch.tensor(scale, device=self.device)

    def update(self):
        forces_b = self.asset._external_force_b[:, self.body_ids]
        forces_b /= self.denom
        self.forces_b = forces_b

    def compute(self) -> torch.Tensor:
        return self.forces_b.reshape(self.num_envs, -1)

    def symmetry_transform(self):
        return cartesian_space_symmetry(self.asset, self.body_names)


class external_torques(Observation):
    supported_backends = ("isaac",)
    def __init__(self, env, body_names, divide_by_mass: bool=True, scale: float = 0.2):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.torques_b = torch.zeros(self.num_envs, len(self.body_ids) * 3, device=self.device)
        default_inertia = self.asset.data.default_inertia[0, 0, [0, 4, 8]].to(self.device)
        self.denom = default_inertia if divide_by_mass else torch.tensor(scale, device=self.device)
    
    def update(self):
        torques_b = self.asset._external_torque_b[:, self.body_ids]
        torques_b = torques_b / self.denom
        self.torques_b = torques_b
    
    def compute(self) -> torch.Tensor:
        return self.torques_b.reshape(self.num_envs, -1)

    def symmetry_transform(self):
        return cartesian_space_symmetry(self.asset, self.body_names, sign=(-1, 1, -1))


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
        self.scan_pos_b = torch.stack([xx, yy, torch.zeros_like(xx)], dim=-1).to(self.device)
        self.shape = self.scan_pos_b.shape[:2]
        
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
        root_pos_w = self.asset.data.root_com_pos_w.reshape(self.num_envs, 1, 1, 3)
        root_quat = yaw_quat(self.asset.data.root_link_quat_w).reshape(self.num_envs, 1, 1, 4)
        
        self.scan_pos_w = root_pos_w + quat_rotate(root_quat, self.scan_pos_b.unsqueeze(0))
        self.height_map_w = self.env.get_ground_height_at(self.scan_pos_w)
        
        height_map = (root_pos_w[:, :, :, 2] - self.height_map_w).clamp(-1., 1.)
        if self.include_xy:
            xy = einops.rearrange(self.scan_pos_b[..., :2], "X Y C -> C X Y")
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
            pos = self.scan_pos_w.clone()
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
            self.asset.data.root_link_quat_w.unsqueeze(1),
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


class raycast_camera(Observation):
    def __init__(
        self,
        env,
        width: int,
        height: int,
        fov: float,
        body_name: Optional[str] = None,
        near: float = 0.01,
        far: float = 100.0,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.near, self.far = near, far

        self.raymap = raymap(width, height, fov).to(self.device)
        self.shape = self.raymap.shape[:2]
        self.num_rays = self.raymap.shape[0] * self.raymap.shape[1]
        self.ground_mesh = self.env.ground_mesh

        if body_name is not None:
            self.body_id = self.asset.find_bodies(body_name)[0]
            assert len(self.body_id) == 1, f"Multiple bodies found for name {body_name}"
            self.body_id = self.body_id[0]
        else:
            self.body_id = None

        if self.env.backend == "isaac" and self.env.sim.has_gui():
            self.marker = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path=f"/Visuals/Command/raycast_camera",
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
        if self.body_id is not None:
            ray_starts = self.asset.data.body_link_pos_w[:, self.body_id]
            quat = self.asset.data.body_link_quat_w[:, self.body_id]
        else:
            ray_starts = self.asset.data.root_pos_w
            quat = self.asset.data.root_link_quat_w
        ray_dirs = quat_rotate(quat.unsqueeze(1), self.raymap.reshape(1, self.num_rays, 3))
        ray_starts = ray_starts.unsqueeze(1) + ray_dirs * self.near
        
        self.ray_starts_w = ray_starts
        self.ray_dirs_w = ray_dirs

        _, ray_distance, _, _ = raycast_mesh(
            ray_starts=ray_starts,
            ray_directions=ray_dirs,
            max_dist=self.far,
            mesh=self.ground_mesh,
            return_distance=True,
        )
        ray_distance = ray_distance.nan_to_num(posinf=self.far)
        self.ray_hits_w = ray_starts + ray_distance.reshape(self.num_envs, self.num_rays, 1) * ray_dirs
        return ray_distance.reshape(self.num_envs, 1, self.shape[0], self.shape[1])
    
    def debug_draw(self) -> None:
        if self.env.backend == "isaac":
            pos = self.ray_hits_w[0].reshape(-1, 3)
            self.marker.visualize(pos)
            # self.env.debug_draw.vector(
            #     self.ray_starts_w[0].reshape(-1, 3),
            #     self.ray_dirs_w[0].reshape(-1, 3),
            #     color=(0.8, 0.0, 0.8, 1.0),
            # )


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
#         self.feet_pos_w = self.asset.data.body_link_pos_w[:, self.body_ids]
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
