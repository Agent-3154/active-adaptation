"""Lambert RGB-D scene cameras (:class:`simple_raycaster.RaycastCamera`).

Shared :class:`LambertRaycastCameraSensor` instances own renderer state (intrinsics,
mesh binding, lighting). Observation terms mount virtual cameras and call
:meth:`LambertRaycastCameraSensor.render` with explicit poses.

Mesh geometry and poses come from the scene-owned
:class:`~active_adaptation.envs.mesh_registry.MeshRegistry`.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from active_adaptation.envs.mesh_registry import MeshRegistry
from active_adaptation.registry import Registry
from active_adaptation.utils.math import quat_from_euler_xyz, quat_mul, quat_rotate

registry = Registry.instance()

# Body / user mount is +X forward, +Y left, +Z up. RaycastCamera + Viser use
# OpenCV / ROS optical (+Z forward, +Y down, +X right).
_MOUNT_TO_OPENCV = (0.5, -0.5, 0.5, -0.5)

# ``mesh_simplify_factor``: scalar (all targets) or per-target map.
MeshSimplifyFactor = float | Mapping[str, float]


def _normalize_simplify_factors(
    spec: MeshSimplifyFactor | None,
    targets: Sequence[str],
) -> dict[str, float]:
    """Return ``{target: factor}`` for targets with ``factor > 0``."""
    if spec is None:
        return {}
    if isinstance(spec, Mapping):
        out: dict[str, float] = {}
        unknown = []
        target_set = set(targets)
        for key, value in spec.items():
            name = str(key)
            factor = float(value)
            if name not in target_set:
                unknown.append(name)
                continue
            if factor > 0.0:
                out[name] = factor
        if unknown:
            raise ValueError(
                f"mesh_simplify_factor keys {unknown} are not in sensor targets "
                f"{list(targets)}"
            )
        return out

    factor = float(spec)
    if factor <= 0.0:
        return {}
    return {str(t): factor for t in targets}


class LambertRaycastCameraSensor:
    """Shared Lambert RGB-D renderer (geometry + intrinsics, no fixed mount)."""

    def __init__(
        self,
        *,
        name: str,
        resolution: Sequence[int] = (64, 48),
        fov_y_deg: float = 70.0,
        near: float = 0.05,
        far: float = 50.0,
        targets: Sequence[str] = ("terrain",),
        light_dir: Sequence[float] = (0.45, -0.35, 0.82),
        ambient: float = 0.30,
        diffuse: float = 0.80,
        closest_hit: bool = True,
        mesh_simplify_factor: MeshSimplifyFactor = 0.0,
    ) -> None:
        self.name = name
        self.width = int(resolution[0])
        self.height = int(resolution[1])
        self.fov_y_deg = float(fov_y_deg)
        self.near = float(near)
        self.far = float(far)
        self.targets = tuple(str(t) for t in targets)
        self.light_dir = tuple(float(x) for x in light_dir)
        self.ambient = float(ambient)
        self.diffuse = float(diffuse)
        self.closest_hit = bool(closest_hit)
        # Scalar or ``{target: factor}``. 0 / omitted target = no simplify.
        # Camera-local only (does not mutate shared ``MeshRegistry`` Warp meshes).
        self.mesh_simplify_factor = _normalize_simplify_factors(
            mesh_simplify_factor, self.targets
        )

        self._camera = None
        self._mesh_registry: MeshRegistry | None = None
        self._mesh_indices: tuple[int, ...] = ()

    def initialize(self, env: Any) -> None:
        from simple_raycaster import RaycastCamera

        scene = env.scene
        self.device = env.device
        self.num_envs = env.num_envs
        if not self.targets:
            raise ValueError(f"Sensor {self.name!r}: targets must be non-empty")

        mesh_registry = MeshRegistry.for_scene(
            scene, backend=env.backend, device=self.device
        )
        self._mesh_indices = mesh_registry.ensure_targets(scene, self.targets)
        self._mesh_registry = mesh_registry

        self.mount2opencv = torch.tensor(_MOUNT_TO_OPENCV, device=self.device)
        self._camera = RaycastCamera(
            self.width,
            self.height,
            fov_y_deg=self.fov_y_deg,
            near=self.near,
            far=self.far,
            convention="opencv",
            device=self.device,
            ambient=self.ambient,
            diffuse=self.diffuse,
            light_dir=self.light_dir,
            outputs=("rgb", "depth"),
            closest_hit=self.closest_hit,
        )
        self._bind_camera_meshes(mesh_registry)
        self._camera.initialize()
        self.update(0.0)

    def _bind_camera_meshes(self, mesh_registry: MeshRegistry) -> None:
        """Bind registry meshes; optionally quadric-decimate selected targets."""
        assert self._camera is not None
        factors = self.mesh_simplify_factor
        if not factors:
            self._camera.bind_meshes(mesh_registry.geometry_view(self._mesh_indices))
            return

        # Per-target simplify → must upload camera-local trimeshes (mixed factors).
        simplified = []
        for target in self.targets:
            idxs = mesh_registry.target_indices[target]
            factor = factors.get(target, 0.0)
            for mesh in mesh_registry.trimeshes_for_indices(idxs):
                if factor > 0.0:
                    m = mesh.copy()
                    m = m.simplify_quadric_decimation(factor)
                    simplified.append(m)
                else:
                    simplified.append(mesh)
        self._camera.bind_trimeshes(simplified)

    def update(self, dt: float | None = None) -> None:
        """Refresh cached mesh poses from the scene (called from ``scene.update``)."""
        del dt
        if self._mesh_registry is not None:
            self._mesh_registry.update_poses(self.num_envs)

    @torch.no_grad()
    def render(
        self,
        cam_pos_w: torch.Tensor,
        cam_quat_w: torch.Tensor,
        clone: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render RGB-D for explicit camera poses → ``rgb, depth, mask``.

        Note that the underlying camera reuses ouput buffers, set `clone` to `True` to get a copy of the tensors.
        """
        camera = self._camera
        mesh_registry = self._mesh_registry
        assert camera is not None and mesh_registry is not None
        mesh_pos_w, mesh_quat_w = mesh_registry.poses_for_indices(self._mesh_indices)
        tensors = camera.render(
            cam_pos_w,
            cam_quat_w,
            mesh_pos_w=mesh_pos_w,
            mesh_quat_w=mesh_quat_w,
            light_dir=self.light_dir,
        )
        if clone:
            tensors = tuple(tensor.clone() for tensor in tensors)
        return tensors

    def mount_pose(
        self,
        entity: Any,
        body_id: int,
        offset_pos: torch.Tensor,
        offset_rpy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Body mount + FLU offset → OpenCV ``(cam_pos_w, cam_quat_w)``."""
        body_pos_w = entity.data.body_link_pos_w[:, body_id]
        body_quat_w = entity.data.body_link_quat_w[:, body_id]
        offset_pos_t = offset_pos
        mount_quat = quat_mul(quat_from_euler_xyz(offset_rpy), self.mount2opencv)
        offset_w = quat_rotate(body_quat_w, offset_pos_t.unsqueeze(0))
        cam_pos_w = body_pos_w + offset_w
        cam_quat_w = quat_mul(body_quat_w, mount_quat.expand(self.num_envs, 4))
        return cam_pos_w, cam_quat_w

    def has_debug_vis_implementation(self) -> bool:
        return False


def lambert_raycast_camera(
    backend: str,
    name: str,
    *,
    resolution: Sequence[int] = (64, 48),
    fov_y_deg: float = 70.0,
    near: float = 0.05,
    far: float = 50.0,
    targets: Sequence[str] = ("terrain",),
    light_dir: Sequence[float] = (0.45, -0.35, 0.82),
    ambient: float = 0.30,
    diffuse: float = 0.80,
    closest_hit: bool = True,
    mesh_simplify_factor: MeshSimplifyFactor = 0.0,
) -> LambertRaycastCameraSensor:
    """Build a shared Lambert RGB-D renderer (no mount — obs terms call ``render``).

    Task YAML under ``sensors.<name>``::

        sensors:
          shared_camera:
            _target_: lambert_raycast_camera
            resolution: [128, 96]
            fov_y_deg: 70.0
            targets: [terrain, robot]
            # closest_hit: false
            # mesh_simplify_factor: 0.5              # all targets
            # mesh_simplify_factor: {robot: 0.5}     # selective (skip terrain / boxes)

    Pair with :class:`~active_adaptation.envs.mdp.observations.extero.camera.raycast_camera`
    observation terms that supply mount ``body_name`` / ``offset_*``.
    """
    del backend
    return LambertRaycastCameraSensor(
        name=name,
        resolution=resolution,
        fov_y_deg=fov_y_deg,
        near=near,
        far=far,
        targets=targets,
        light_dir=light_dir,
        ambient=ambient,
        diffuse=diffuse,
        closest_hit=closest_hit,
        mesh_simplify_factor=mesh_simplify_factor,
    )


registry.register("sensor", "lambert_raycast_camera", lambert_raycast_camera)
