from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Optional, Tuple

import torch
from typing_extensions import override

import active_adaptation
from ..base import Observation
from active_adaptation.utils.math import (
    quat_rotate,
    quat_rotate_inverse,
    yaw_quat,
)
from active_adaptation.utils.symmetry import SymmetryTransform, cartesian_space_symmetry

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from active_adaptation.envs.env_base import _EnvBase

if active_adaptation.get_backend() == "isaaclab":
    from isaaclab.utils.warp import raycast_mesh

from simple_raycaster import MultiMeshRaycaster


class external_forces(Observation):
    supported_backends = ("isaaclab",)

    def __init__(self, body_names, divide_by_mass: bool = True, scale: float = 1.0):
        super().__init__()
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


class external_torques(Observation):
    supported_backends = ("isaaclab",)

    def __init__(self, body_names, divide_by_mass: bool = True, scale: float = 0.2):
        super().__init__()
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


class height_scan(Observation):
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
        super().__init__()
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
            if self.env.backend == "isaaclab":
                from isaacsim.core.utils.stage import get_current_stage

                stage = get_current_stage()
                for target in self.targets:
                    target_asset = self.env.scene[target]
                    prim_path = target_asset.root_physx_view.prim_paths[0]
                    self.raycaster.add_from_path(prim_path, stage=stage)
                    self.target_assets.append(target_asset)
            else:
                raise NotImplementedError(f"Unsupported backend: {self.env.backend}")

        if self.env.backend == "isaaclab" and self.env.sim.has_gui():
            from active_adaptation.envs.backends.isaaclab import IsaacSceneAdapter

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

        hit_pos_w, _, _ = self.raycaster.raycast_fused(
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
        if self.env.backend == "isaaclab":
            self.marker.visualize(self.hit_pos_w.reshape(-1, 3))

    def symmetry_transform(self):
        if self.flatten:
            perm = torch.arange(self.shape.numel()).reshape(self.shape).flip((1,)).reshape(-1)
            signs = torch.ones(self.shape.numel())
        else:
            perm = torch.arange(self.shape[1]).flip(0)
            signs = torch.ones(self.shape[1])
        return SymmetryTransform(perm=perm, signs=signs)


class forward_scan(Observation):
    supported_backends = ("isaaclab",)

    def __init__(
        self,
        hfov: Tuple[float, float],
        vfov: Tuple[float, float],
        resolution: Tuple[int, int],
        max_range: float = 5.0,
        flatten: bool = False,
    ):
        super().__init__()
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

        if self.env.backend == "isaaclab" and self.env.sim.has_gui():
            from active_adaptation.envs.backends.isaaclab import IsaacSceneAdapter

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
        if self.env.backend == "isaaclab":
            pos = self.ray_hits.reshape(-1, 3)
            self.marker.visualize(pos)


class feet_height_map(Observation):
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
        super().__init__()
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

        if self.env.backend == "isaaclab" and self.env.sim.has_gui():
            from active_adaptation.envs.backends.isaaclab import IsaacSceneAdapter

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
        if self.env.backend == "isaaclab":
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


class closest_points(Observation):
    """Closest surface points on target meshes from selected robot bodies.

    Probe positions are the link origins of ``body_names``. Each name may be a
    regex (Isaac ``find_bodies``). ``targets`` are scene entity keys whose
    visuals are registered with :class:`~simple_raycaster.MeshProximitySensor`.

    ``clipping_range=(near, far)`` sets the query radius to ``far``. Hits closer
    than ``near`` are clamped to ``near``. Misses (no surface within ``far``)
    report ``far`` when ``distance_only``, else a zero vector.

    Returns:

    * ``distance_only=True``: clamped distances ``[N, n_bodies]``.
    * ``distance_only=False``: flattened closest-point positions
      ``[N, n_bodies * 3]`` —
      * ``frame="body"``: each point in its body frame
        (``R_bodyᵀ (p* − p_body)``).
      * ``frame="root"``: each point relative to the robot root in the root
        frame (``R_rootᵀ (p* − p_root)``).
    """

    supported_backends = ("isaaclab",)

    def __init__(
        self,
        body_names: str | List[str],
        clipping_range: Tuple[float, float],
        targets: List[str],
        frame: Literal["root", "body"] = "body",
        distance_only: bool = False,
    ) -> None:
        super().__init__()
        if frame not in ("root", "body"):
            raise ValueError(f"frame must be 'root' or 'body', got {frame!r}")
        near, far = float(clipping_range[0]), float(clipping_range[1])
        if far <= near:
            raise ValueError(f"clipping_range far ({far}) must be > near ({near})")
        self.body_names_cfg = body_names
        self.clipping_range = (near, far)
        self.targets = list(targets)
        self.frame = frame
        self.distance_only = distance_only

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(
            self.body_names_cfg, preserve_order=True
        )
        if len(self.body_ids) == 0:
            raise ValueError(f"No bodies matched {self.body_names_cfg!r}")
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.num_bodies = len(self.body_ids)
        self.near, self.far = self.clipping_range

        from simple_raycaster import MeshProximitySensor

        self.sensor = MeshProximitySensor(device=self.device)
        if len(self.targets) == 0:
            raise ValueError("closest_points requires at least one target entity")
        for target in self.targets:
            self.sensor.add_isaac_entity(self.env.scene[target])

        if self.env.backend == "isaaclab" and self.env.sim.has_gui():
            from active_adaptation.envs.backends.isaaclab import IsaacSceneAdapter

            scene: IsaacSceneAdapter = self.env.scene
            self.marker_query = scene.create_sphere_marker(
                "/Visuals/Command/closest_points_query",
                (0.2, 0.8, 0.2),
                radius=0.01,
            )
            self.marker_hit = scene.create_sphere_marker(
                "/Visuals/Command/closest_points_hit",
                (0.9, 0.2, 0.1),
                radius=0.01,
            )

    @override
    def compute(self) -> torch.Tensor:
        body_pos_w = self.asset.data.body_link_pos_w[:, self.body_ids]  # [N, B, 3]
        closest_w, dist = self.sensor.query(body_pos_w, max_dist=self.far)
        self.body_pos_w = body_pos_w
        self.closest_pos_w = closest_w
        self.distances = dist

        dist_c = dist.clamp(self.near, self.far)
        if self.distance_only:
            return dist_c

        hit = dist < self.far
        # Length-clamp hits into [near, far]; misses stay zero after masking.
        length = (closest_w - body_pos_w).norm(dim=-1).clamp_min(1e-8)
        closest_w = body_pos_w + (closest_w - body_pos_w) * (dist_c / length).unsqueeze(-1)
        closest_w = torch.where(hit.unsqueeze(-1), closest_w, body_pos_w)

        displacement = closest_w - body_pos_w
        if self.frame == "body":
            body_quat_w = self.asset.data.body_link_quat_w[:, self.body_ids]
            closest_f = quat_rotate_inverse(body_quat_w, displacement)
        else:
            root_quat_w = self.asset.data.root_link_quat_w.unsqueeze(1)
            closest_f = quat_rotate_inverse(root_quat_w, displacement)

        return closest_f.reshape(self.num_envs, -1)

    def debug_draw(self) -> None:
        if self.env.backend == "isaaclab" and hasattr(self, "marker_query"):
            self.marker_query.visualize(self.body_pos_w[0].reshape(-1, 3))
            hit = self.distances[0] < self.far
            if hit.any():
                self.marker_hit.visualize(self.closest_pos_w[0][hit].reshape(-1, 3))

    @override
    def symmetry_transform(self) -> SymmetryTransform:
        if self.distance_only:
            return cartesian_space_symmetry(self.asset, self.body_names, sign=(1,))
        if self.frame == "body":
            raise NotImplementedError("Symmetry transform is not implemented for frame=body and distance_only=False")
        return cartesian_space_symmetry(self.asset, self.body_names)
