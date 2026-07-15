from __future__ import annotations

import colorsys
import math
import torch
import einops
from typing import TYPE_CHECKING, List, Optional, Tuple

from typing_extensions import override

import active_adaptation
from jaxtyping import Float
from .base import ObservationV2
from active_adaptation.utils.math import (
    quat_rotate,
    yaw_quat,
    quat_from_euler_xyz,
    root_pose_from_view_z_up,
)
from active_adaptation.utils.symmetry import SymmetryTransform, cartesian_space_symmetry

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.sensors import TiledCamera
    from isaaclab.scene import InteractiveSceneCfg
    from mjlab.sensor import CameraSensor
    from mjlab.scene import SceneCfg
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


class camera_isaac(ObservationV2):
    """Isaac Lab tiled camera observation for the Isaac backend.

    Registers a :class:`~isaaclab.sensors.TiledCameraCfg` on the scene during
    :meth:`edit_spec` (called from :meth:`IsaacBackendEnv.setup_scene` before
    the scene is built). After simulation startup, :meth:`compute` reads
    ``camera.data.output[data_type]``, optionally normalizes it, and returns a
    ``(num_envs, C, H, W)`` tensor.

    Args:
        resolution: ``(width, height)`` in pixels.
        data_type: Camera output key, e.g. ``"rgb"`` or ``"depth"``.
        focal_length: Pinhole focal length in mm.
        focus_distance: Pinhole focus distance in m.
        horizontal_aperture: Pinhole horizontal aperture in mm.
        clipping_range: Near/far clipping planes in m.
        body_name: If set, attach the camera to ``Robot/{body_name}``; otherwise
            spawn a standalone camera under each env namespace.
        sensor_name: Scene sensor attribute name. Defaults to a unique
            ``tiled_camera_{id}`` per instance so multiple cameras can coexist.
        offset_pos: Camera offset translation w.r.t. its parent frame.
        offset_rot: Camera offset rotation ``(w, x, y, z)`` w.r.t. parent frame.
        offset_convention: Offset frame convention (``"ros"``, ``"world"``, or
            ``"opengl"``).
    """

    supported_backends = ("isaac",)
    _instance_count = 0

    def __init__(
        self,
        resolution: Tuple[int, int],
        data_type: str = "rgb",
        focal_length: float = 24.0,
        focus_distance: float = 400.0,
        horizontal_aperture: float = 20.955,
        clipping_range: Tuple[float, float] = (0.1, 20.0),
        body_name: Optional[str] = None,
        sensor_name: Optional[str] = None,
        offset_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        offset_rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        offset_convention: str = "world",
    ):
        super().__init__()
        self.resolution = resolution
        self.data_type = data_type
        self.focal_length = focal_length
        self.focus_distance = focus_distance
        self.horizontal_aperture = horizontal_aperture
        self.clipping_range = tuple(clipping_range)
        self.body_name = body_name
        self.offset_pos = tuple(offset_pos)
        self.offset_rot = tuple(offset_rot)
        self.offset_convention = offset_convention

        if sensor_name is None:
            camera_isaac._instance_count += 1
            self.sensor_name = f"tiled_camera_{camera_isaac._instance_count}"
        else:
            self.sensor_name = sensor_name

    @override
    def edit_spec(self, scene_config: InteractiveSceneCfg) -> None:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObjectCfg
        from isaaclab.sensors import TiledCameraCfg

        if hasattr(scene_config, self.sensor_name):
            raise ValueError(
                f"Scene config already has sensor '{self.sensor_name}'. "
                "Choose a distinct sensor_name for each camera_isaac instance."
            )

        if self.body_name is not None:
            camera_mount_cfg = None
            prim_path = f"{{ENV_REGEX_NS}}/Robot/{self.body_name}/{self.sensor_name}"
        else:
            # As of Isaac Sim 5.1.0, there is a bug that prevents setting the pose
            # of TiledCamera dynamically during simulation using `set_world_poses`
            # Therefore we attach it to a dummy body
            # TODO@btx0424: check if this is still needed after Isaac Sim 6.0.0
            camera_mount_cfg = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{self.sensor_name}_mount",
                spawn=sim_utils.SphereCfg(
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        kinematic_enabled=True,
                    ),
                    radius=0.02,
                )
            )
            prim_path = f"{{ENV_REGEX_NS}}/{self.sensor_name}_mount/{self.sensor_name}"

        cfg = TiledCameraCfg(
            prim_path=prim_path,
            offset=TiledCameraCfg.OffsetCfg(
                pos=self.offset_pos,
                rot=self.offset_rot,
                convention=self.offset_convention,
            ),
            data_types=[self.data_type],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=self.focal_length,
                focus_distance=self.focus_distance,
                horizontal_aperture=self.horizontal_aperture,
                clipping_range=self.clipping_range,
            ),
            width=self.resolution[0],
            height=self.resolution[1],
            update_latest_camera_pose=True,
        )
        if camera_mount_cfg is not None:
            setattr(scene_config, self.sensor_name + "_mount", camera_mount_cfg)
        setattr(scene_config, self.sensor_name, cfg)

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.env.render_enabled = True
        if self.body_name is None:
            self.camera_mount: RigidObject = self.env.scene.entities[self.sensor_name + "_mount"]
        self.camera: TiledCamera = self.env.scene.sensors[self.sensor_name]

    @override
    def update(self) -> None:
        if self.body_name is None:
            robot_root_pos_w = self.env.scene.entities["robot"].data.root_link_pos_w
            eye = robot_root_pos_w + torch.tensor([2.0, 2.0, 2.0], device=self.device)
            target = robot_root_pos_w
            self.camera_mount.write_root_link_pose_to_sim(
                root_pose_from_view_z_up(eye, target)
            )

    @override
    def compute(self) -> torch.Tensor:
        # for rgb, isaac sim returns uint8, which we leave as is
        data = self.camera.data.output[self.data_type]  # NHWC        
        # Temporary in-line code to save some images for sanity check
        # from pathlib import Path
        # from torchvision.utils import make_grid, save_image
        # if not hasattr(self, "_cam_dbg_step"):
        #     self._cam_dbg_step = 0
        # if self._cam_dbg_step < 100:
        #     raw = self.camera.data.output[self.data_type][:16].detach()
        #     if self.data_type == "rgb":
        #         imgs = raw.permute(0, 3, 1, 2).float() / 255.0
        #     else:
        #         imgs = raw.permute(0, 3, 1, 2).clone()
        #         imgs[torch.isinf(imgs)] = 0.0
        #         imgs = imgs / imgs.max().clamp_min(1e-6)
        #         if imgs.shape[1] == 1:
        #             imgs = imgs.expand(-1, 3, -1, -1)
        #     out_dir = Path("test_images")
        #     out_dir.mkdir(parents=True, exist_ok=True)
        #     save_image(
        #         make_grid(imgs, nrow=max(1, round(imgs.shape[0] ** 0.5))),
        #         out_dir / f"{self.sensor_name}_{self._cam_dbg_step:04d}.png",
        #     )
        # self._cam_dbg_step += 1
        return einops.rearrange(data, "n h w c -> n c h w")

    @override
    def symmetry_transform(self) -> SymmetryTransform:
        if self.body_name is None:
            raise NotImplementedError("Symmetry transform is only available when the camera is attached to a body")
        width = self.resolution[0]
        perm = torch.arange(width - 1, -1, -1, dtype=torch.long)
        return SymmetryTransform(perm, torch.ones(width))


class camera_mjlab(ObservationV2):
    """MjLab camera observation using :class:`~mjlab.sensor.CameraSensor`.

    Registers a :class:`~mjlab.sensor.CameraSensorCfg` during scene construction
    (``edit_spec`` appends to the mjlab sensor list). Rendering runs via
    ``sim.sense()`` each env step; :meth:`compute` returns ``(N, C, H, W)``.

    MuJoCo cameras use **fixed** mode: ``pos``/``quat`` are set in the spec at
    build time. Attach via ``body_name`` (``robot/{name}``) or spawn on the
    worldbody with ``offset_pos`` / ``offset_rot``. Dynamic tracking (e.g. via a
    dummy mocap body) is not implemented yet.

    Args:
        resolution: ``(width, height)`` in pixels.
        data_type: ``"rgb"``, ``"depth"``, or ``"segmentation"``.
        fovy: Vertical field of view in degrees. If ``None``, derived from
            ``focal_length`` and ``horizontal_aperture``.
        focal_length: Pinhole focal length in mm (used when ``fovy`` is None).
        horizontal_aperture: Pinhole horizontal aperture in mm.
        body_name: Robot body to attach the camera to (``robot/{name}``).
        sensor_name: Scene sensor key. Defaults to ``mjlab_camera_{id}``.
        offset_pos: Camera position w.r.t. parent body or world frame.
        offset_rot: Camera orientation ``(w, x, y, z)`` w.r.t. parent frame.
    """

    supported_backends = ("mjlab",)
    _instance_count = 0

    def __init__(
        self,
        resolution: Tuple[int, int],
        data_type: str = "rgb",
        fovy: Optional[float] = None,
        focal_length: float = 24.0,
        horizontal_aperture: float = 20.955,
        body_name: Optional[str] = None,
        sensor_name: Optional[str] = None,
        offset_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        offset_rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ):
        super().__init__()
        self.resolution = resolution
        self.data_type = data_type
        self.fovy = fovy
        self.focal_length = focal_length
        self.horizontal_aperture = horizontal_aperture
        self.body_name = body_name
        self.offset_pos = tuple(offset_pos)
        self.offset_rot = tuple(offset_rot)

        if sensor_name is None:
            camera_mjlab._instance_count += 1
            self.sensor_name = f"mjlab_camera_{camera_mjlab._instance_count}"
        else:
            self.sensor_name = sensor_name

    def _resolved_fovy(self) -> float:
        if self.fovy is not None:
            return self.fovy
        width, height = self.resolution
        h_fov = 2.0 * math.atan(self.horizontal_aperture / (2.0 * self.focal_length))
        v_fov = 2.0 * math.atan(math.tan(h_fov / 2.0) * height / width)
        return math.degrees(v_fov)

    @override
    def edit_spec(self, scene_config: SceneCfg) -> None:
        from mjlab.sensor import CameraSensorCfg
        parent_body = f"robot/{self.body_name}" if self.body_name is not None else None
        scene_config.sensors += (
            CameraSensorCfg(
                name=self.sensor_name,
                parent_body=parent_body,
                pos=self.offset_pos,
                quat=self.offset_rot,
                fovy=self._resolved_fovy(),
                width=self.resolution[0],
                height=self.resolution[1],
                data_types=(self.data_type,),
            ),
        )

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.camera: CameraSensor = self.env.scene.sensors[self.sensor_name]

    @override
    def compute(self) -> torch.Tensor:
        data = self.camera.data
        if self.data_type == "rgb":
            output = data.rgb
        elif self.data_type == "depth":
            output = data.depth.clone()
            output[torch.isinf(output)] = 0.0
        elif self.data_type == "segmentation":
            output = data.segmentation
        else:
            raise ValueError(f"Unsupported camera data_type: {self.data_type!r}")

        # Temporary in-line code to save some images for sanity check
        # from pathlib import Path
        # from torchvision.utils import make_grid, save_image
        # if not hasattr(self, "_cam_dbg_step"):
        #     self._cam_dbg_step = 0
        # if self._cam_dbg_step < 100:
        #     raw = output[:16].detach()
        #     if self.data_type == "rgb":
        #         imgs = raw.permute(0, 3, 1, 2).float() / 255.0
        #     else:
        #         imgs = raw.permute(0, 3, 1, 2).clone()
        #         imgs[torch.isinf(imgs)] = 0.0
        #         imgs = imgs / imgs.max().clamp_min(1e-6)
        #         if imgs.shape[1] == 1:
        #             imgs = imgs.expand(-1, 3, -1, -1)
        #     out_dir = Path("test_images")
        #     out_dir.mkdir(parents=True, exist_ok=True)
        #     save_image(
        #         make_grid(imgs, nrow=max(1, round(imgs.shape[0] ** 0.5))),
        #         out_dir / f"{self.sensor_name}_{self._cam_dbg_step:04d}.png",
        #     )
        # self._cam_dbg_step += 1
        return einops.rearrange(output, "n h w c -> n c h w")

    @override
    def symmetry_transform(self) -> SymmetryTransform:
        if self.body_name is None:
            raise NotImplementedError(
                "camera_mjlab symmetry is only defined for body-mounted cameras"
            )
        width = self.resolution[0]
        perm = torch.arange(width - 1, -1, -1, dtype=torch.long)
        return SymmetryTransform(perm, torch.ones(width))