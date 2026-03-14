import torch
import numpy as np
import einops
from typing import Tuple, TYPE_CHECKING, Optional, List

import active_adaptation
from jaxtyping import Float
from .base import Observation
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse, yaw_quat, quat_from_euler_xyz
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
    that pixel. The rays are in camera space, where +X is forward, +Y is left, and +Z is up.
    
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
        self.asset: Articulation = self.env.scene.articulations["robot"]
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
        self.asset: Articulation = self.env.scene.articulations["robot"]
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
    """
    Ground height sampled on a 2D grid in the robot's horizontal plane via downward raycasts.

    Builds a grid over (x_range × y_range) at the given resolution, casts rays straight down
    from each grid point (in world frame, transformed by the robot's pose), and records
    hit height. The observation is the height relative to the robot root (root_z - hit_z),
    with optional additive Gaussian noise (noise_scale) and clamping to clamp_range.

    The raycaster uses the env ground mesh by default; pass ``targets`` (scene asset names)
    to also raycast against additional meshes (e.g. obstacles). Output shape is either
    (num_envs, n_points) when ``flatten=True`` or (num_envs, 1, H, W) when ``flatten=False``.
    """

    def __init__(
        self,
        env,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        resolution: Tuple[float, float],
        flatten: bool=False,
        noise_scale = 0.02,
        clamp_range: Tuple[float, float] = (-1., 1.),
        targets: Optional[List[str]] = None,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.flatten = flatten
        self.noise_scale = noise_scale
        self.clamp_range = clamp_range

        with torch.device(self.device):
            x = torch.linspace(x_range[0], x_range[1], int((x_range[1] - x_range[0]) / resolution[0])+1)
            y = torch.linspace(y_range[0], y_range[1], int((y_range[1] - y_range[0]) / resolution[1])+1)
            xx, yy = torch.meshgrid(x, y, indexing="ij")
            self.scan_pos_b = torch.stack([xx, yy, torch.zeros_like(xx)], dim=-1).to(self.device)
            self.shape = self.scan_pos_b.shape[:2]
            self.n_rays = self.shape.numel()

            self.ground_mesh_pos_w = torch.tensor([0., 0., 0.,]).expand(self.num_envs, 1, 3)
            self.ground_mesh_quat_w = torch.tensor([1., 0., 0., 0.]).expand(self.num_envs, 1, 4)
            self.ray_dirs_w = torch.tensor([0., 0., -1.]).expand(self.num_envs, self.n_rays, 3)

        from simple_raycaster import MultiMeshRaycaster
        self.raycaster = MultiMeshRaycaster([self.env.ground_mesh], device=self.device)
        self.target_assets = []
        
        if targets is not None:
            if self.env.backend == "isaac":
                from isaacsim.core.utils.stage import get_current_stage
                stage = get_current_stage()
                for target in targets:
                    target_asset = self.env.scene[target]
                    prim_path = target_asset.root_physx_view.prim_paths[0]
                    self.raycaster.add_from_path(prim_path, stage=stage)
                    self.target_assets.append(target_asset)
            else:
                raise NotImplementedError(f"Unsupported backend: {self.env.backend}")
        
        if self.env.backend == "isaac" and self.env.sim.has_gui():
            from active_adaptation.envs.backends.isaac import IsaacSceneAdapter
            scene: IsaacSceneAdapter = self.env.scene
            self.marker = scene.create_sphere_marker(
                "/Visuals/Command/height_scan", (0.8, 0.0, 0.0), radius=0.02
            )

    def compute(self):
        root_pos_w = self.asset.data.root_com_pos_w.reshape(self.num_envs, 1, 1, 3)
        root_quat = yaw_quat(self.asset.data.root_link_quat_w).reshape(self.num_envs, 1, 1, 4)
        
        self.scan_pos_w = (
            root_pos_w
            + torch.tensor([0., 0., 10.], device=self.device)
            + quat_rotate(root_quat, self.scan_pos_b.unsqueeze(0))
        )
        
        if len(self.target_assets) > 0:
            mesh_pos_w = torch.stack([self.ground_mesh_pos_w] + [target_asset.data.root_link_pos_w for target_asset in self.target_assets], dim=1)
            mesh_quat_w = torch.stack([self.ground_mesh_quat_w] + [target_asset.data.root_link_quat_w for target_asset in self.target_assets], dim=1)
        else:
            mesh_pos_w = self.ground_mesh_pos_w
            mesh_quat_w = self.ground_mesh_quat_w
        
        hit_pos_w, _ = self.raycaster.raycast_fused(
            mesh_pos_w=mesh_pos_w,
            mesh_quat_w=mesh_quat_w,
            ray_starts_w=self.scan_pos_w.reshape(self.num_envs, self.n_rays, 3),
            ray_dirs_w=self.ray_dirs_w,
        )
        self.hit_pos_w = hit_pos_w.reshape(self.num_envs, *self.shape, 3) # [N, X, Y, 3]
        
        height_map = root_pos_w[:, :, :, 2] - self.hit_pos_w[:, :, :, 2]
        height_map = (height_map + self.noise_scale * torch.randn_like(height_map)).clamp(*self.clamp_range)
        if self.flatten:
            return height_map.reshape(self.num_envs, -1)
        else:
            return height_map.reshape(self.num_envs, -1, *self.shape)
    
    def debug_draw(self):
        if self.env.backend == "isaac":
            self.marker.visualize(self.hit_pos_w.reshape(-1, 3))

    def symmetry_transform(self):
        if self.flatten:
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
        self.asset: Articulation = self.env.scene.articulations["robot"]
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
            from active_adaptation.envs.backends.isaac import IsaacSceneAdapter
            scene: IsaacSceneAdapter = self.env.scene
            self.marker = scene.create_sphere_marker(
                "/Visuals/Command/forward_scan", (0.8, 0.0, 0.0), radius=0.02
            )
    
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
    supported_dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        # "uint16": torch.uint16,
        # "uint8": torch.uint8,
    }

    def __init__(
        self,
        env,
        resolution: Tuple[int, int],
        fov: float,
        rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        body_name: Optional[str] = None,
        near: float = 0.01,
        far: float = 100.0,
        dtype: torch.dtype | str = torch.float16,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.near, self.far = near, far
        self.dtype = self.supported_dtypes[dtype] if isinstance(dtype, str) else dtype
        assert (
            self.dtype in self.supported_dtypes.values()
        ), f"Unsupported dtype: {dtype}"
        assert self.far - self.near > 1e-6, "Far must be greater than near"

        width, height = resolution
        self.raymap = raymap(width, height, fov).to(self.device)
        quat = quat_from_euler_xyz(torch.tensor(rpy, device=self.device))
        self.raymap = quat_rotate(quat.reshape(1, 1, 4), self.raymap)
        
        self.shape = self.raymap.shape[:2]
        assert self.shape == (height, width), "Resolution must match the raymap shape"
        self.num_rays = self.raymap.shape[0] * self.raymap.shape[1]
        self.ground_mesh = self.env.ground_mesh

        if body_name is not None:
            self.body_id = self.asset.find_bodies(body_name)[0]
            assert len(self.body_id) == 1, f"Multiple bodies found for name {body_name}"
            self.body_id = self.body_id[0]
        else:
            self.body_id = None

        if self.env.backend == "isaac" and self.env.sim.has_gui():
            from active_adaptation.envs.backends.isaac import IsaacSceneAdapter
            scene: IsaacSceneAdapter = self.env.scene
            self.marker = scene.create_sphere_marker(
                "/Visuals/Command/raycast_camera", (0.8, 0.0, 0.8), radius=0.02
            )
    
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
        self.ray_hits_w = ray_starts + ray_distance.reshape(self.num_envs, self.num_rays, 1) * ray_dirs
        
        ray_distance = ray_distance.nan_to_num(posinf=self.far)
        # Convert to target dtype
        if self.dtype.is_floating_point:
            # For float32 and float16, direct conversion
            ray_distance = ray_distance.to(self.dtype)
        else:
            # For uint16 and uint8, normalize to [0, 1] and scale to dtype max value
            range_size = self.far - self.near
            normalized = (ray_distance - self.near) / range_size
            max_val = torch.iinfo(self.dtype).max
            ray_distance = (normalized * max_val).clamp(0, max_val).to(self.dtype)
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


class feet_height_map(Observation):
    """
    Per-foot local height map around each contact point.

    For every foot body, a small pattern of rays is cast downward around the
    foot position. The observation is the difference between the foot height
    and the ground hit height for each ray, normalized by ``nomial_height``.

    This can be used as a compact exteroceptive signal indicating whether
    terrain around each foot is higher or lower than a nominal height.
    """

    def __init__(
        self,
        env,
        feet_names: str = ".*_foot",
        nomial_height: float = 0.3,
        size: float = 0.3,
        clamp_range: Tuple[float, float] = (-1., 1.),
        flatten: bool = True,
    ):
        super().__init__(env)
        # Store configuration
        self.nominal_height = nomial_height
        self.clamp_range = clamp_range
        self.flatten = flatten

        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(feet_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.num_feet = len(self.body_ids)

        # Ray start pattern in a small square around the origin, expressed in
        # the foot's yaw-aligned local frame and then shifted above the feet.
        # z=10 is arbitrary; only relative height differences matter.
        xx = torch.linspace(-size/2, size/2, 3, device=self.device)
        yy = torch.linspace(-size/2, size/2, 3, device=self.device)
        xx, yy = torch.meshgrid(xx, yy, indexing="ij")
        self.ray_starts = torch.stack([xx, yy, torch.zeros_like(xx)], dim=-1).reshape(-1, 3)
        self.num_rays = len(self.ray_starts)

        if self.env.backend == "isaac" and self.env.sim.has_gui():
            from active_adaptation.envs.backends.isaac import IsaacSceneAdapter
            scene: IsaacSceneAdapter = self.env.scene
            self.marker = scene.create_sphere_marker(
                "/Visuals/Command/feet_height_map", (0.8, 0.0, 0.8), radius=0.02
            )
        
    def compute(self) -> torch.Tensor:
        """
        Return normalized per-foot height map.

        The map is flattened over feet and ray samples and divided by the
        nominal height scale to keep values in a reasonable range.
        """
        feet_pos_w = self.asset.data.body_link_pos_w[:, self.body_ids]
        quat = yaw_quat(self.asset.data.root_link_quat_w)

        # Compute ray start positions in world frame for each foot and ray.
        expand_shape = (self.num_envs, self.num_feet, self.num_rays, 3)
        ray_starts = self.ray_starts.reshape(1, 1, -1, 3).expand(expand_shape)
        query_points = quat_rotate(
            quat.reshape(self.num_envs, 1, 1, 4),
            ray_starts,
        )
        query_points += feet_pos_w.reshape(self.num_envs, self.num_feet, 1, 3)
        ground_height = self.env.get_ground_height_at(query_points)
        
        feet_height = feet_pos_w[:, :, 2:3] - ground_height # [N, F, 1] - [N, F, R]
        feet_height = feet_height.clamp(*self.clamp_range) / self.nominal_height

        self.vis_points = query_points.clone() # [N, F, R, 3]
        self.vis_points[..., 2] = ground_height

        if self.flatten:
            return feet_height.reshape(self.num_envs, -1) # [N, F * R]
        else:
            return feet_height # [N, F, R]
    
    def debug_draw(self):
        if self.env.backend == "isaac":
            self.marker.visualize(self.vis_points.reshape(-1, 3))
    
    def symmetry_transform(self):
        if self.flatten:
            # Base foot-level symmetry (swaps left/right feet using spatial_symmetry_mapping)
            base = cartesian_space_symmetry(self.asset, self.body_names, sign=(1,))
            num_feet = len(self.body_ids)
            num_rays = self.num_rays
            # Per-foot patch permutation: mirror across sagittal plane (y -> -y)
            patch_perm = torch.arange(num_rays).reshape(3, 3).flip(1).reshape(-1)
            # Expand foot-level and patch-level permutations to flattened layout [feet, rays]
            foot_src = base.perm.repeat_interleave(num_rays)
            ray_src = patch_perm.repeat(num_feet)
            perm = foot_src * num_rays + ray_src
            signs = torch.ones_like(perm, dtype=torch.float32)
            x = torch.arange(9).reshape(1, 1, 3, 3)
            x = x + torch.arange(num_feet).reshape(1, num_feet, 1, 1)
            y = x[:, base.perm].flip(3)
            assert torch.all(y.reshape(1, -1) == x.reshape(1, -1)[..., perm])
            return SymmetryTransform(perm=perm, signs=signs)
        else:
            pass
        
