import colorsys
import torch
from typing import TYPE_CHECKING, List, Optional, Tuple

from typing_extensions import override

import active_adaptation
from jaxtyping import Float
from .base import ObservationV2
from active_adaptation.utils.math import quat_rotate, yaw_quat, quat_from_euler_xyz
from active_adaptation.utils.symmetry import SymmetryTransform, cartesian_space_symmetry

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from active_adaptation.envs.env_base import _EnvBase

if active_adaptation.get_backend() == "isaac":
    from isaaclab.utils.warp import raycast_mesh


# from simple_raycaster import MultiMeshRaycaster


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
    u = torch.arange(width, dtype=torch.float32)
    v = torch.arange(height, dtype=torch.float32)

    uu, vv = torch.meshgrid(u, v, indexing="xy")

    u_ndc = (uu + 0.5) / width * 2.0 - 1.0
    v_ndc = 1.0 - (vv + 0.5) / height * 2.0

    aspect_ratio = width / height

    tan_fov_half = torch.tan(torch.tensor(fov / 2.0))
    u_camera = u_ndc * tan_fov_half
    v_camera = v_ndc * tan_fov_half / aspect_ratio

    x_camera = torch.ones_like(u_camera)
    directions = torch.stack([x_camera, v_camera, u_camera], dim=-1)

    directions = directions / directions.norm(dim=-1, keepdim=True)

    return directions


def _distinct_debug_color(instance_id: int) -> Tuple[float, float, float]:
    """Pick a saturated, high-contrast RGB color for debug markers."""
    hue = (instance_id * 0.618033988749895) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.85, 0.95)


class external_forces(ObservationV2):
    supported_backends = ("isaac",)

    def __init__(self, body_names, divide_by_mass: bool = True, scale: float = 1.0):
        self.body_names_pattern = body_names
        self.divide_by_mass = divide_by_mass
        self.scale = scale

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(self.body_names_pattern)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.forces_b = torch.zeros(self.num_envs, len(self.body_ids) * 3, device=self.device)
        default_mass_total = self.asset.data.default_mass[0].sum() * 9.81
        self.denom = default_mass_total if self.divide_by_mass else torch.tensor(
            self.scale, device=self.device
        )

    def update(self):
        forces_b = self.asset._external_force_b[:, self.body_ids]
        forces_b /= self.denom
        self.forces_b = forces_b

    def compute(self) -> torch.Tensor:
        return self.forces_b.reshape(self.num_envs, -1)

    def symmetry_transform(self):
        return cartesian_space_symmetry(self.asset, self.body_names)


class external_torques(ObservationV2):
    supported_backends = ("isaac",)

    def __init__(self, body_names, divide_by_mass: bool = True, scale: float = 0.2):
        self.body_names_pattern = body_names
        self.divide_by_mass = divide_by_mass
        self.scale = scale

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(self.body_names_pattern)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.torques_b = torch.zeros(self.num_envs, len(self.body_ids) * 3, device=self.device)
        default_inertia = self.asset.data.default_inertia[0, 0, [0, 4, 8]].to(self.device)
        self.denom = default_inertia if self.divide_by_mass else torch.tensor(
            self.scale, device=self.device
        )

    def update(self):
        torques_b = self.asset._external_torque_b[:, self.body_ids]
        torques_b = torques_b / self.denom
        self.torques_b = torques_b

    def compute(self) -> torch.Tensor:
        return self.torques_b.reshape(self.num_envs, -1)

    def symmetry_transform(self):
        return cartesian_space_symmetry(self.asset, self.body_names, sign=(-1, 1, -1))


class height_scan(ObservationV2):
    """
    Ground height sampled on a 2D grid in the robot's horizontal plane via downward raycasts.
    """

    def __init__(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        resolution: Tuple[float, float],
        flatten: bool = False,
        noise_scale=0.02,
        clamp_range: Tuple[float, float] = (-1.0, 1.0),
        targets: Optional[List[str]] = None,
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        self.flatten = flatten
        self.noise_scale = noise_scale
        self.clamp_range = clamp_range
        self.targets = targets

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]

        with torch.device(self.device):
            x = torch.linspace(
                self.x_range[0],
                self.x_range[1],
                int((self.x_range[1] - self.x_range[0]) / self.resolution[0]) + 1,
            )
            y = torch.linspace(
                self.y_range[0],
                self.y_range[1],
                int((self.y_range[1] - self.y_range[0]) / self.resolution[1]) + 1,
            )
            xx, yy = torch.meshgrid(x, y, indexing="ij")
            self.scan_pos_b = torch.stack([xx, yy, torch.zeros_like(xx)], dim=-1).to(self.device)
            self.shape = self.scan_pos_b.shape[:2]
            self.n_rays = self.shape.numel()

            self.ground_mesh_pos_w = torch.tensor([0.0, 0.0, 0.0]).expand(self.num_envs, 1, 3)
            self.ground_mesh_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(self.num_envs, 1, 4)
            self.ray_dirs_w = torch.tensor([0.0, 0.0, -1.0]).expand(self.num_envs, self.n_rays, 3)

        self.raycaster = MultiMeshRaycaster([self.env.ground_mesh], device=self.device)
        self.target_assets = []

        if self.targets is not None:
            if self.env.backend == "isaac":
                from isaacsim.core.utils.stage import get_current_stage

                stage = get_current_stage()
                for target in self.targets:
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
            + torch.tensor([0.0, 0.0, 10.0], device=self.device)
            + quat_rotate(root_quat, self.scan_pos_b.unsqueeze(0))
        )

        if len(self.target_assets) > 0:
            mesh_pos_w = torch.cat(
                [self.ground_mesh_pos_w]
                + [target_asset.data.root_link_pos_w.unsqueeze(1) for target_asset in self.target_assets],
                dim=1,
            )
            mesh_quat_w = torch.cat(
                [self.ground_mesh_quat_w]
                + [target_asset.data.root_link_quat_w.unsqueeze(1) for target_asset in self.target_assets],
                dim=1,
            )
        else:
            mesh_pos_w = self.ground_mesh_pos_w
            mesh_quat_w = self.ground_mesh_quat_w

        hit_pos_w, _ = self.raycaster.raycast_fused(
            mesh_pos_w=mesh_pos_w,
            mesh_quat_w=mesh_quat_w,
            ray_starts_w=self.scan_pos_w.reshape(self.num_envs, self.n_rays, 3),
            ray_dirs_w=self.ray_dirs_w,
        )
        self.hit_pos_w = hit_pos_w.reshape(self.num_envs, *self.shape, 3)

        height_map = root_pos_w[:, :, :, 2] - self.hit_pos_w[:, :, :, 2]
        height_map = (height_map + self.noise_scale * torch.randn_like(height_map)).clamp(
            *self.clamp_range
        )
        if self.flatten:
            return height_map.reshape(self.num_envs, -1)
        return height_map.reshape(self.num_envs, -1, *self.shape)

    def debug_draw(self):
        if self.env.backend == "isaac":
            self.marker.visualize(self.hit_pos_w.reshape(-1, 3))

    def symmetry_transform(self):
        if self.flatten:
            perm = torch.arange(self.shape.numel()).reshape(self.shape).flip((1,)).reshape(-1)
            signs = torch.ones(self.shape.numel())
        else:
            perm = torch.arange(self.shape[1]).flip(0)
            signs = torch.ones(self.shape[1])
        return SymmetryTransform(perm=perm, signs=signs)


class forward_scan(ObservationV2):
    supported_backends = ("isaac",)

    def __init__(
        self,
        hfov: Tuple[float, float],
        vfov: Tuple[float, float],
        resolution: Tuple[int, int],
        max_range: float = 5.0,
        flatten: bool = False,
    ):
        self.hfov = hfov
        self.vfov = vfov
        self.resolution = resolution
        self.max_range = max_range
        self.flatten = flatten

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.ground_mesh = self.env.ground_mesh

        hangles = torch.linspace(self.hfov[0], self.hfov[1], self.resolution[0])
        vangles = torch.linspace(self.vfov[0], self.vfov[1], self.resolution[1])
        vv, hh = torch.meshgrid(vangles, hangles, indexing="ij")
        directions = torch.stack(
            [
                torch.cos(hh) * torch.cos(vv),
                torch.sin(hh) * torch.cos(vv),
                torch.sin(vv),
            ],
            dim=-1,
        )
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
            self.directions.expand(self.num_envs, self.num_rays, 3),
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
        return ray_distance.reshape(self.num_envs, 1, *self.shape)

    def symmetry_transform(self):
        if self.flatten:
            perm = torch.arange(self.shape.numel())
            perm = perm.reshape(self.shape).flip(1)
            return SymmetryTransform(perm=perm.reshape(-1), signs=torch.ones(perm.numel()))
        return SymmetryTransform(
            perm=torch.arange(self.shape[1]).flip(0),
            signs=torch.ones(self.shape[1]),
        )

    def debug_draw(self):
        if self.env.backend == "isaac":
            pos = self.ray_hits.reshape(-1, 3)
            self.marker.visualize(pos)


class raycast_camera(ObservationV2):
    supported_backends = ("isaac",)
    _debug_instance_count = 0

    supported_dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
    }

    def __init__(
        self,
        resolution: Tuple[int, int],
        fov_deg: float,
        rpy_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        pos_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        body_name: Optional[str] = None,
        near: float = 0.01,
        far: float = 100.0,
        dtype: torch.dtype | str = torch.float16,
        targets: Optional[List[str]] = None,
    ):
        self.resolution = resolution
        self.fov_deg = fov_deg
        self.rpy_deg = rpy_deg
        self.pos_offset = pos_offset
        self.body_name = body_name
        self.near = near
        self.far = far
        self.dtype = dtype
        self.targets = targets

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.dtype = (
            self.supported_dtypes[self.dtype] if isinstance(self.dtype, str) else self.dtype
        )
        assert self.dtype in self.supported_dtypes.values(), f"Unsupported dtype: {self.dtype}"
        assert self.far - self.near > 1e-6, "Far must be greater than near"

        width, height = self.resolution
        self.raymap = raymap(width, height, self.fov_deg / 180.0 * torch.pi).to(self.device)
        euler = torch.tensor(self.rpy_deg, device=self.device) / 180.0 * torch.pi
        quat = quat_from_euler_xyz(euler)
        self.raymap = quat_rotate(quat.reshape(1, 1, 4), self.raymap)
        self.pos_offset = torch.tensor(self.pos_offset, device=self.device)

        self.shape = self.raymap.shape[:2]
        assert self.shape == (height, width), "Resolution must match the raymap shape"
        self.num_rays = self.raymap.shape[0] * self.raymap.shape[1]

        from simple_raycaster import MultiMeshRaycasterV2

        self.raycaster = MultiMeshRaycasterV2(device=self.device)
        self.raycaster.add_isaac_static("/World/ground")
        if self.targets is not None:
            for target in self.targets:
                target_asset = self.env.scene[target]
                self.raycaster.add_isaac_entity(target_asset)

        if self.body_name is not None:
            self.body_id = self.asset.find_bodies(self.body_name)[0]
            assert len(self.body_id) == 1, f"Multiple bodies found for name {self.body_name}"
            self.body_id = self.body_id[0]
        else:
            self.body_id = None

        if self.env.backend == "isaac" and self.env.sim.has_gui():
            from active_adaptation.envs.backends.isaac import IsaacSceneAdapter

            scene: IsaacSceneAdapter = self.env.scene
            self.instance_id = raycast_camera._debug_instance_count
            raycast_camera._debug_instance_count += 1
            marker_color = _distinct_debug_color(self.instance_id)
            self.marker = scene.create_sphere_marker(
                f"/Visuals/Command/raycast_camera_{self.instance_id}",
                marker_color,
                radius=0.02,
            )

    def compute(self) -> torch.Tensor:
        if self.body_id is not None:
            body_pos_w = self.asset.data.body_link_pos_w[:, self.body_id]
            body_quat = self.asset.data.body_link_quat_w[:, self.body_id]
        else:
            body_pos_w = self.asset.data.root_link_pos_w
            body_quat = self.asset.data.root_link_quat_w
        self.ray_dirs_w = quat_rotate(
            body_quat.unsqueeze(1), self.raymap.reshape(1, self.num_rays, 3)
        )
        offset_w = quat_rotate(body_quat, self.pos_offset.unsqueeze(0))
        self.ray_starts_w = (
            body_pos_w.reshape(self.num_envs, 1, 3)
            + offset_w.reshape(self.num_envs, 1, 3)
            + self.ray_dirs_w * self.near
        )

        hit_pos_w, hit_distance = self.raycaster.raycast_fused(
            ray_starts_w=self.ray_starts_w,
            ray_dirs_w=self.ray_dirs_w,
            min_dist=0.0,
            max_dist=self.far,
        )
        self.ray_hits_w = hit_pos_w

        hit_distance = hit_distance.nan_to_num(posinf=self.far).to(self.dtype)
        return hit_distance.reshape(self.num_envs, 1, self.shape[0], self.shape[1])

    def debug_draw(self) -> None:
        if self.env.backend == "isaac":
            pos = self.ray_hits_w[0].reshape(-1, 3)
            self.marker.visualize(pos)

    def symmetry_transform(self):
        perm = torch.arange(self.shape[1]).flip(0)
        signs = torch.ones(self.shape[1])
        x = torch.arange(self.shape[0] * self.shape[1]).reshape(1, 1, *self.shape)
        y = x.flip(3)
        assert torch.all(y == x[..., perm]), "raycast_camera symmetry permutation mismatch"
        return SymmetryTransform(perm=perm, signs=signs)


class feet_height_map(ObservationV2):
    """
    Per-foot local height map around each contact point.
    """

    def __init__(
        self,
        feet_names: str = ".*_foot",
        nomial_height: float = 0.3,
        size: float = 0.3,
        clamp_range: Tuple[float, float] = (-1.0, 1.0),
        flatten: bool = True,
    ):
        self.feet_names_pattern = feet_names
        self.nominal_height = nomial_height
        self.size = size
        self.clamp_range = clamp_range
        self.flatten = flatten

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(self.feet_names_pattern)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.num_feet = len(self.body_ids)

        xx = torch.linspace(-self.size / 2, self.size / 2, 3, device=self.device)
        yy = torch.linspace(-self.size / 2, self.size / 2, 3, device=self.device)
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
        feet_pos_w = self.asset.data.body_link_pos_w[:, self.body_ids]
        quat = yaw_quat(self.asset.data.root_link_quat_w)

        expand_shape = (self.num_envs, self.num_feet, self.num_rays, 3)
        ray_starts = self.ray_starts.reshape(1, 1, -1, 3).expand(expand_shape)
        query_points = quat_rotate(quat.reshape(self.num_envs, 1, 1, 4), ray_starts)
        query_points += feet_pos_w.reshape(self.num_envs, self.num_feet, 1, 3)
        ground_height = self.env.get_ground_height_at(query_points)

        feet_height = feet_pos_w[:, :, 2:3] - ground_height
        feet_height = feet_height.clamp(*self.clamp_range) / self.nominal_height

        self.vis_points = query_points.clone()
        self.vis_points[..., 2] = ground_height

        if self.flatten:
            return feet_height.reshape(self.num_envs, -1)
        return feet_height

    def debug_draw(self):
        if self.env.backend == "isaac":
            self.marker.visualize(self.vis_points.reshape(-1, 3))

    def symmetry_transform(self):
        if self.flatten:
            base = cartesian_space_symmetry(self.asset, self.body_names, sign=(1,))
            num_feet = len(self.body_ids)
            num_rays = self.num_rays
            patch_perm = torch.arange(num_rays).reshape(3, 3).flip(1).reshape(-1)
            foot_src = base.perm.repeat_interleave(num_rays)
            ray_src = patch_perm.repeat(num_feet)
            perm = foot_src * num_rays + ray_src
            signs = torch.ones_like(perm, dtype=torch.float32)
            x = torch.arange(9).reshape(1, 1, 3, 3)
            x = x + torch.arange(num_feet).reshape(1, num_feet, 1, 1)
            y = x[:, base.perm].flip(3)
            assert torch.all(y.reshape(1, -1) == x.reshape(1, -1)[..., perm])
            return SymmetryTransform(perm=perm, signs=signs)
        return None
