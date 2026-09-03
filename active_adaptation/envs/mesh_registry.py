"""Scene-owned Warp mesh geometry and cached world poses.

Centralizes mesh registration and pose refresh for Warp consumers (Lambert
RGB-D sensors, future proximity / lidar terms). Cameras and raycasters in
``simple-raycaster`` remain pose-explicit: they receive ``mesh_pos_w`` /
``mesh_quat_w`` from here, never gather from simulation entities themselves.

Lifecycle::

    registry = MeshRegistry.for_scene(scene, backend=env.backend, device=env.device)
    indices = registry.ensure_targets(scene, ("terrain", "robot"))
    registry.update_poses(num_envs)          # once per control step
    pos, quat = registry.poses_for_indices(indices)

``scene.update()`` calls :meth:`LambertRaycastCameraSensor.update` on warp
sensors, which refreshes cached mesh poses.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import trimesh

_TERRAIN_KEY = "terrain"


@dataclass(frozen=True)
class _MeshBindView:
    """Minimal ``bind_meshes`` target: a ``meshes_wp`` list slice."""

    meshes_wp: list


class MeshRegistry:
    """Cached Warp meshes + world poses for a simulation scene."""

    def __init__(self, scene: Any, *, backend: str, device: str | torch.device) -> None:
        self._scene = scene
        self.backend = str(backend)
        self.device = str(device) if not isinstance(device, str) else device
        self.torch_device = torch.device(self.device)

        self.meshes_wp: list[Any] = []
        self.entities: list[Any | None] = []
        self._entity_body_indices: list[tuple[int, ...] | None] = []
        self._target_indices: dict[str, tuple[int, ...]] = {}

        self.mesh_pos_w: torch.Tensor | None = None
        self.mesh_quat_w: torch.Tensor | None = None

    @classmethod
    def for_scene(
        cls,
        scene: Any,
        *,
        backend: str,
        device: str | torch.device,
    ) -> MeshRegistry:
        registry = getattr(scene, "_mesh_registry", None)
        if registry is None:
            registry = cls(scene, backend=backend, device=device)
            scene._mesh_registry = registry
        return registry

    @property
    def n_meshes(self) -> int:
        return len(self.meshes_wp)

    def ensure_targets(self, scene: Any, targets: Sequence[str]) -> tuple[int, ...]:
        """Register ``targets`` if missing; return flat mesh indices for them."""
        if not targets:
            raise ValueError("ensure_targets: targets must be non-empty")
        indices: list[int] = []
        for target in targets:
            key = str(target)
            if key not in self._target_indices:
                self._target_indices[key] = self._register_target(scene, key)
            indices.extend(self._target_indices[key])
        return tuple(indices)

    def geometry_view(self, indices: Sequence[int]) -> _MeshBindView:
        """Return a bind handle for a mesh subset (for :meth:`RaycastCamera.bind_meshes`)."""
        idx = [int(i) for i in indices]
        return _MeshBindView([self.meshes_wp[i] for i in idx])

    def update_poses(self, num_envs: int) -> None:
        """Refresh :attr:`mesh_pos_w` / :attr:`mesh_quat_w` from entity state."""
        n = int(num_envs)
        m = self.n_meshes
        device = self.torch_device
        if m == 0:
            self.mesh_pos_w = torch.zeros(n, 0, 3, device=device)
            self.mesh_quat_w = torch.zeros(n, 0, 4, device=device)
            return

        zero_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        if not self.entities:
            self.mesh_pos_w = torch.zeros(n, m, 3, device=device)
            self.mesh_quat_w = zero_quat.view(1, 1, 4).expand(n, m, 4).contiguous()
            return

        mesh_pos_w: list[torch.Tensor] = []
        mesh_quat_w: list[torch.Tensor] = []
        for entity, body_indices in zip(self.entities, self._entity_body_indices):
            if entity is None:
                mesh_pos_w.append(torch.zeros(n, 1, 3, device=device))
                mesh_quat_w.append(zero_quat.view(1, 1, 4).expand(n, 1, 4))
            else:
                assert body_indices is not None
                body_link_pose_w = entity.data.body_link_pose_w[:, body_indices]
                if body_link_pose_w.shape[0] != n:
                    raise ValueError(
                        f"MeshRegistry batch {n} != entity instance count "
                        f"{body_link_pose_w.shape[0]}"
                    )
                mesh_pos_w.append(body_link_pose_w[:, :, :3])
                mesh_quat_w.append(body_link_pose_w[:, :, 3:7])

        self.mesh_pos_w = torch.cat(mesh_pos_w, dim=1).contiguous()
        self.mesh_quat_w = torch.cat(mesh_quat_w, dim=1).contiguous()

    def poses_for_indices(
        self, indices: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice cached poses for a consumer's mesh subset."""
        if self.mesh_pos_w is None or self.mesh_quat_w is None:
            raise RuntimeError("MeshRegistry.update_poses must run before reading poses")
        idx = list(indices)
        return self.mesh_pos_w[:, idx].contiguous(), self.mesh_quat_w[:, idx].contiguous()

    def _register_target(self, scene: Any, target: str) -> tuple[int, ...]:
        start = self.n_meshes
        if target == _TERRAIN_KEY:
            self._register_terrain(scene)
        else:
            self._register_entity(scene, target)
        end = self.n_meshes
        if end == start:
            raise ValueError(f"target {target!r} registered zero meshes")
        return tuple(range(start, end))

    def _register_terrain(self, scene: Any) -> None:
        if self.backend == "isaaclab":
            self.meshes_wp.append(scene.ground_mesh)
        else:
            from simple_raycaster.helpers import trimesh2wp

            mesh = self._terrain_trimesh(scene)
            self.meshes_wp.append(trimesh2wp(mesh, self.device))
        self.entities.append(None)
        self._entity_body_indices.append(None)

    def _register_entity(self, scene: Any, target: str) -> None:
        from simple_raycaster.helpers import trimesh2wp

        entity = scene.entities[target]
        body_indices, body_names, meshes = scene.get_visual_meshes(target)
        if not meshes:
            raise ValueError(
                f"Entity {target!r} has no non-empty visual meshes for warp rendering"
            )
        if not (len(body_indices) == len(body_names) == len(meshes)):
            raise ValueError(
                f"get_visual_meshes({target!r}) returned mismatched lengths: "
                f"indices={len(body_indices)}, names={len(body_names)}, "
                f"meshes={len(meshes)}"
            )
        kept_indices: list[int] = []
        for body_i, body_name, mesh in zip(body_indices, body_names, meshes):
            if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                warnings.warn(
                    f"MeshRegistry: skipping empty visual mesh for {target!r} "
                    f"body {body_name!r}.",
                    stacklevel=2,
                )
                continue
            self.meshes_wp.append(trimesh2wp(mesh, self.device))
            kept_indices.append(int(body_i))
        if not kept_indices:
            raise ValueError(
                f"Entity {target!r} has no non-empty visual meshes for warp rendering"
            )
        self.entities.append(entity)
        self._entity_body_indices.append(tuple(kept_indices))

    @staticmethod
    def _terrain_trimesh(scene: Any) -> trimesh.Trimesh:
        terrain = scene.entities.get(_TERRAIN_KEY)
        if terrain is None:
            raise KeyError(
                f"terrain target requires a terrain entity "
                f"(have {sorted(scene.entities)})"
            )
        _body_indices, _body_names, meshes = scene.get_collision_meshes(_TERRAIN_KEY)
        kept = [m for m in meshes if len(m.vertices) > 0 and len(m.faces) > 0]
        if not kept:
            raise ValueError("terrain entity has no collision meshes")
        return trimesh.util.concatenate(kept) if len(kept) > 1 else kept[0]
