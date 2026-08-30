"""Warp-backed Lambert RGB-D scene cameras (:class:`simple_raycaster.RaycastCamera`).

These sensors use the megakernel camera in ``raycast_camera.py``. They are **not**
the ``mesh_rgbd`` / ``make_mesh_rgbd_renderer`` path used for 3DGS compositing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch
import trimesh

from active_adaptation.registry import Registry
from active_adaptation.utils.math import quat_from_euler_xyz, quat_mul, quat_rotate

from .warp_base import WarpSensorSpec

registry = Registry.instance()

_TERRAIN_KEY = "terrain"
_ISAAC_GROUND_PRIM = "/World/ground"

# Body / user mount is +X forward, +Y left, +Z up. RaycastCamera + Viser use
# OpenCV / ROS optical (+Z forward, +Y down, +X right). Same quat as
# ``gs_camera._MOUNT_TO_OPENCV`` / ``raycast_camera`` frustum ``ros_to_cam``.
_MOUNT_TO_OPENCV = (0.5, -0.5, 0.5, -0.5)


class _MeshPoseRegistry(Protocol):
    meshes_wp: list
    initialized: bool

    def initialize(self) -> None: ...

    def _get_mesh_poses_w(
        self, n: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class _MjlabMeshRegistry:
    """Minimal ``MultiMeshRaycasterV2``-shaped registry for mjlab scenes."""

    def __init__(self, device: str | torch.device) -> None:
        self.entities: list[Any | None] = []
        self.meshes_wp = []
        self.device = str(device) if not isinstance(device, str) else device
        self.initialized = False
        self.meshes_array = None

    @property
    def n_meshes(self) -> int:
        return len(self.meshes_wp)

    def _add_mesh(self, mesh_wp: Any) -> None:
        self.meshes_wp.append(mesh_wp)
        self.initialized = False

    def add_static_trimesh(self, mesh: trimesh.Trimesh) -> None:
        from simple_raycaster.helpers import trimesh2wp

        self._add_mesh(trimesh2wp(mesh, self.device))
        self.entities.append(None)

    def add_entity(self, entity: Any, meshes: Sequence[trimesh.Trimesh]) -> None:
        from simple_raycaster.helpers import trimesh2wp

        if len(meshes) != entity.num_bodies:
            raise ValueError(
                f"Expected {entity.num_bodies} body meshes, got {len(meshes)}"
            )
        for mesh in meshes:
            if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                raise ValueError(
                    f"Entity {getattr(entity, 'name', entity)!r} has an empty body mesh; "
                    "all bodies must have visual geometry for RaycastCamera"
                )
            self._add_mesh(trimesh2wp(mesh, self.device))
        self.entities.append(entity)

    def initialize(self) -> None:
        if self.initialized:
            return
        import warp as wp

        wp.init()
        self.meshes_array = wp.array(
            [mesh.id for mesh in self.meshes_wp],
            device=self.device,
            dtype=wp.uint64,
        )
        self.initialized = True

    def _get_mesh_poses_w(
        self, n: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.n_meshes == 0:
            raise ValueError("No meshes registered")

        zero_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        if not self.entities:
            mesh_pos_w = torch.zeros(n, self.n_meshes, 3, device=device)
            mesh_quat_w = zero_quat.expand(n, self.n_meshes, 4)
            return mesh_pos_w, mesh_quat_w

        mesh_pos_w = []
        mesh_quat_w = []
        for entity in self.entities:
            if entity is None:
                mesh_pos_w.append(torch.zeros(n, 1, 3, device=device))
                mesh_quat_w.append(zero_quat.expand(n, 1, 4))
            else:
                body_link_pose_w = entity.data.body_link_pose_w
                if body_link_pose_w.shape[0] != n:
                    raise ValueError(
                        f"Batch size {n} != entity instance count "
                        f"{body_link_pose_w.shape[0]}"
                    )
                mesh_pos_w.append(body_link_pose_w[:, :, :3])
                mesh_quat_w.append(body_link_pose_w[:, :, 3:7])

        return torch.cat(mesh_pos_w, dim=1), torch.cat(mesh_quat_w, dim=1)


@dataclass
class LambertRaycastCameraData:
    """Latest render products from :class:`LambertRaycastCameraSensor`."""

    rgb: torch.Tensor
    depth: torch.Tensor
    mask: torch.Tensor


class LambertRaycastCameraSensor:
    """Scene-owned :class:`~simple_raycaster.RaycastCamera` Lambert RGB-D sensor."""

    def __init__(
        self,
        *,
        env: Any,
        name: str,
        entity: str = "robot",
        body_name: str | None = None,
        pattern: str | None = None,
        offset_pos: Sequence[float] = (0.0, 0.0, 0.0),
        offset_rpy: Sequence[float] = (0.0, 0.0, 0.0),
        resolution: Sequence[int] = (64, 48),
        fov_y_deg: float = 70.0,
        near: float = 0.05,
        far: float = 50.0,
        targets: Sequence[str] = (_TERRAIN_KEY,),
        light_dir: Sequence[float] = (0.45, -0.35, 0.82),
        ambient: float = 0.30,
        diffuse: float = 0.80,
    ) -> None:
        self.env = env
        self.name = name
        self.entity_name = entity
        self.body_name = body_name
        self.pattern = pattern
        self.offset_pos = tuple(float(x) for x in offset_pos)
        self.offset_rpy = tuple(float(x) for x in offset_rpy)
        self.width = int(resolution[0])
        self.height = int(resolution[1])
        self.fov_y_deg = float(fov_y_deg)
        self.near = float(near)
        self.far = float(far)
        self.targets = tuple(str(t) for t in targets)
        self.light_dir = tuple(float(x) for x in light_dir)
        self.ambient = float(ambient)
        self.diffuse = float(diffuse)

        self.device = env.device
        self.num_envs = env.num_envs
        self.data = LambertRaycastCameraData(
            rgb=torch.zeros(self.num_envs, self.height, self.width, 3, device=self.device),
            depth=torch.zeros(self.num_envs, self.height, self.width, device=self.device),
            mask=torch.zeros(self.num_envs, self.height, self.width, device=self.device),
        )
        self._camera = None
        self._mesh_registry: _MeshPoseRegistry | None = None
        self._mount_entity = None
        self._body_id: int | None = None
        self._offset_pos_t: torch.Tensor | None = None
        self._mount_quat: torch.Tensor | None = None
        self.cam_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.cam_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.cam_quat_w[:, 0] = 1.0

    def initialize(self) -> None:
        from simple_raycaster import RaycastCamera

        scene = self.env.scene
        mount = scene.entities[self.entity_name]
        self._mount_entity = mount

        if self.body_name is not None:
            body_ids, body_names = mount.find_bodies(self.body_name)
        elif self.pattern is not None:
            body_ids, body_names = mount.find_bodies(self.pattern)
        else:
            body_ids, body_names = mount.find_bodies(mount.body_names[0])
        if len(body_ids) != 1:
            raise ValueError(
                f"Sensor {self.name!r}: expected one mount body on {self.entity_name!r}, "
                f"got {body_names}"
            )
        self._body_id = int(body_ids[0])

        euler = torch.tensor(self.offset_rpy, device=self.device) * (torch.pi / 180.0)
        # User offset in body FLU, then FLU → OpenCV so identity looks +X forward.
        self._mount_quat = quat_mul(
            quat_from_euler_xyz(euler).reshape(1, 4),
            torch.tensor(_MOUNT_TO_OPENCV, device=self.device),
        )
        self._offset_pos_t = torch.tensor(self.offset_pos, device=self.device)

        if not self.targets:
            raise ValueError(f"Sensor {self.name!r}: targets must be non-empty")

        self._mesh_registry = self._build_mesh_registry(scene)
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
        )
        self._camera.bind_meshes(self._mesh_registry)
        self._camera.initialize()

    def _build_mesh_registry(self, scene: Any) -> _MeshPoseRegistry:
        if self.env.backend in ("isaaclab", "isaaclab"):
            from simple_raycaster import MultiMeshRaycasterV2

            registry = MultiMeshRaycasterV2(device=self.device)
            for target in self.targets:
                if target == _TERRAIN_KEY:
                    registry.add_isaac_static(_ISAAC_GROUND_PRIM)
                else:
                    registry.add_isaac_entity(scene.entities[target])
            return registry

        registry = _MjlabMeshRegistry(self.device)
        for target in self.targets:
            if target == _TERRAIN_KEY:
                registry.add_static_trimesh(self._terrain_trimesh(scene))
            else:
                meshes = scene.get_visual_meshes(target)
                registry.add_entity(scene.entities[target], meshes)
        return registry

    @staticmethod
    def _terrain_trimesh(scene: Any) -> trimesh.Trimesh:
        terrain = scene.entities.get(_TERRAIN_KEY)
        if terrain is None:
            raise KeyError(
                f"terrain target requires a terrain entity "
                f"(have {sorted(scene.entities)})"
            )
        meshes = scene.get_collision_meshes(_TERRAIN_KEY)
        kept = [m for m in meshes if len(m.vertices) > 0 and len(m.faces) > 0]
        if not kept:
            raise ValueError("terrain entity has no collision meshes")
        return trimesh.util.concatenate(kept) if len(kept) > 1 else kept[0]

    @torch.no_grad()
    def update(self) -> None:
        mount = self._mount_entity
        camera = self._camera
        assert mount is not None and self._body_id is not None
        assert camera is not None
        assert self._offset_pos_t is not None and self._mount_quat is not None

        body_pos_w = mount.data.body_link_pos_w[:, self._body_id]
        body_quat_w = mount.data.body_link_quat_w[:, self._body_id]
        offset_w = quat_rotate(body_quat_w, self._offset_pos_t.unsqueeze(0))
        cam_pos_w = body_pos_w + offset_w
        cam_quat_w = quat_mul(body_quat_w, self._mount_quat.expand(self.num_envs, 4))

        rgb, depth, mask = camera.render(
            cam_pos_w,
            cam_quat_w,
            light_dir=self.light_dir,
        )
        self.cam_pos_w = cam_pos_w
        self.cam_quat_w = cam_quat_w
        self.data.rgb = rgb
        self.data.depth = depth
        self.data.mask = mask
    
    def has_debug_vis_implementation(self) -> bool:
        # make isaaclab happy
        return False


def lambert_raycast_camera(
    backend: str,
    name: str,
    *,
    entity: str = "robot",
    body_name: str | None = None,
    pattern: str | None = None,
    offset_pos: Sequence[float] = (0.0, 0.0, 0.0),
    offset_rpy: Sequence[float] = (0.0, 0.0, 0.0),
    resolution: Sequence[int] = (64, 48),
    fov_y_deg: float = 70.0,
    near: float = 0.05,
    far: float = 50.0,
    targets: Sequence[str] = (_TERRAIN_KEY,),
    light_dir: Sequence[float] = (0.45, -0.35, 0.82),
    ambient: float = 0.30,
    diffuse: float = 0.80,
) -> WarpSensorSpec:
    """Build a deferred Lambert RGB-D scene camera.

    Uses :class:`simple_raycaster.RaycastCamera` (``raycast_camera.py`` megakernel).
    This is **not** ``mesh_rgbd.make_mesh_rgbd_renderer`` (that path is for 3DGS
    mesh compositing only).

    Task YAML under ``sensors.<name>``::

        sensors:
          front_rgbd:
            _target_: lambert_raycast_camera
            entity: robot
            body_name: base_link
            resolution: [128, 96]
            fov_y_deg: 70.0
            targets: [terrain, object]

    Mount (``entity``, ``body_name`` / ``pattern``, ``offset_*``) is in the
    body frame: **+X forward, +Y left, +Z up**. Zero ``offset_rpy`` looks
    along body +X. Internally converted to OpenCV for ``RaycastCamera``
    (+Z optical). ``targets`` are scene keys to render against.

    Read ``env.scene.sensors[name].data.{rgb,depth,mask}`` after each step.
    """
    del backend
    return WarpSensorSpec(
        name=name,
        factory=LambertRaycastCameraSensor,
        kwargs={
            "entity": entity,
            "body_name": body_name,
            "pattern": pattern,
            "offset_pos": offset_pos,
            "offset_rpy": offset_rpy,
            "resolution": resolution,
            "fov_y_deg": fov_y_deg,
            "near": near,
            "far": far,
            "targets": targets,
            "light_dir": light_dir,
            "ambient": ambient,
            "diffuse": diffuse,
        },
    )


registry.register("sensor", "lambert_raycast_camera", lambert_raycast_camera)
