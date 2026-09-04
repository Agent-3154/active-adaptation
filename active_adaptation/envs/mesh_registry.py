"""Scene-owned Warp mesh geometry and cached world poses.

Centralizes mesh registration and pose refresh for Warp consumers (Lambert
RGB-D sensors, future proximity / lidar terms). Cameras and raycasters in
``simple-raycaster`` remain pose-explicit: they receive ``mesh_pos_w`` /
``mesh_quat_w`` from here, never gather from simulation entities themselves.

Lifecycle::

    registry = MeshRegistry.for_scene(scene, backend=env.backend, device=env.device)
    indices = registry.ensure_targets(scene, ("terrain", "robot", "pedestal"))
    registry.update_poses(num_envs)          # once per control step
    pos, quat = registry.poses_for_indices(indices)

Targets:

* ``terrain`` — static ground (identity pose; mesh already world-framed).
* ``scene.entities[name]`` — articulations / rigid objects; poses from
  ``body_link_pose_w`` each ``update_poses``.
* ``name/body_regex`` — same as an entity target, but only bodies whose names
  ``re.fullmatch`` the regex (e.g. ``robot/(gripper_.*|base_link)``).
* ``scene.extras[name]`` — Isaac ``AssetBaseCfg`` collision-only props; mesh
  loaded once and world poses snapped once at registration (they do not move).
  **Known issue:** Isaac Lab 2.3.2 ``XformPrimView.get_world_poses()`` may
  return only a single env; poses are currently expanded via ``env_origins``
  (see ``_register_extra``). Replace when the view reports all instances.

``scene.update()`` calls :meth:`LambertRaycastCameraSensor.update` on warp
sensors, which refreshes cached mesh poses.
"""

from __future__ import annotations

import re
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


def parse_mesh_target(target: str) -> tuple[str, str | None]:
    """Split ``entity`` or ``entity/body_regex`` into ``(entity, pattern|None)``."""
    key = str(target)
    if "/" not in key:
        return key, None
    entity, pattern = key.split("/", 1)
    if not entity:
        raise ValueError(f"Invalid mesh target {target!r}: empty entity name")
    if not pattern:
        raise ValueError(f"Invalid mesh target {target!r}: empty body regex after '/'")
    return entity, pattern


def target_entity_name(target: str) -> str:
    """Entity / extra / terrain name for a target (strip optional body regex)."""
    return parse_mesh_target(target)[0]


class MeshRegistry:
    """Cached Warp meshes + world poses for a simulation scene."""

    def __init__(self, scene: Any, *, backend: str, device: str | torch.device) -> None:
        self._scene = scene
        self.backend = str(backend)
        self.device = str(device) if not isinstance(device, str) else device
        self.torch_device = torch.device(self.device)

        self.meshes_wp: list[Any] = []
        # CPU trimeshes parallel to ``meshes_wp`` (for optional camera simplify).
        self.trimeshes: list[trimesh.Trimesh] = []
        # One entry per registered *group* (terrain / entity / extra). A group
        # may contribute multiple meshes (multi-body entities); pose tensors
        # are (N, K, …) with K = number of meshes in that group.
        self.entities: list[Any | None] = []
        self._entity_body_indices: list[tuple[int, ...] | None] = []
        # Fixed world poses for static extras (None → refresh or terrain zeros).
        self._fixed_pos_w: list[torch.Tensor | None] = []
        self._fixed_quat_w: list[torch.Tensor | None] = []
        self._target_indices: dict[str, tuple[int, ...]] = {}

        self.mesh_pos_w: torch.Tensor | None = None
        self.mesh_quat_w: torch.Tensor | None = None

    @property
    def target_indices(self) -> dict[str, tuple[int, ...]]:
        """Registered target name → flat mesh indices into ``meshes_wp`` / ``trimeshes``."""
        return self._target_indices

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
        """Register ``targets`` if missing; return flat mesh indices for them.

        Each target may be ``terrain``, an entity/extra name, or
        ``entity/body_regex`` to keep only matching visual bodies.
        """
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

    def trimeshes_for_indices(self, indices: Sequence[int]) -> list[trimesh.Trimesh]:
        """CPU trimeshes for ``indices`` (same order as :meth:`geometry_view`)."""
        if len(self.trimeshes) != self.n_meshes:
            raise RuntimeError(
                f"MeshRegistry trimeshes ({len(self.trimeshes)}) out of sync with "
                f"meshes_wp ({self.n_meshes})"
            )
        return [self.trimeshes[int(i)] for i in indices]

    def update_poses(self, num_envs: int) -> None:
        """Refresh :attr:`mesh_pos_w` / :attr:`mesh_quat_w` from entity / fixed state."""
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
        for entity, body_indices, fixed_pos, fixed_quat in zip(
            self.entities,
            self._entity_body_indices,
            self._fixed_pos_w,
            self._fixed_quat_w,
        ):
            if fixed_pos is not None:
                assert fixed_quat is not None
                if fixed_pos.shape[0] != n:
                    raise ValueError(
                        f"MeshRegistry batch {n} != fixed-pose instance count "
                        f"{fixed_pos.shape[0]}"
                    )
                mesh_pos_w.append(fixed_pos)
                mesh_quat_w.append(fixed_quat)
            elif entity is None:
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
        entity_name, body_pattern = parse_mesh_target(target)
        if entity_name == _TERRAIN_KEY:
            if body_pattern is not None:
                raise ValueError(
                    f"terrain target does not support a body filter "
                    f"(got {target!r})"
                )
            self._register_terrain(scene)
        elif entity_name in scene.entities:
            self._register_entity(scene, entity_name, body_pattern=body_pattern)
        elif entity_name in (getattr(scene, "extras", None) or {}):
            if body_pattern is not None:
                raise ValueError(
                    f"extra target {entity_name!r} does not support a body filter "
                    f"(got {target!r})"
                )
            self._register_extra(scene, entity_name)
        else:
            extras = getattr(scene, "extras", None) or {}
            raise KeyError(
                f"MeshRegistry target {target!r} (entity {entity_name!r}) not found "
                f"in scene.entities {sorted(scene.entities)} or "
                f"scene.extras {sorted(extras)}"
            )
        end = self.n_meshes
        if end == start:
            raise ValueError(f"target {target!r} registered zero meshes")
        return tuple(range(start, end))

    def _append_group(
        self,
        *,
        entity: Any | None,
        body_indices: tuple[int, ...] | None,
        fixed_pos: torch.Tensor | None = None,
        fixed_quat: torch.Tensor | None = None,
    ) -> None:
        self.entities.append(entity)
        self._entity_body_indices.append(body_indices)
        self._fixed_pos_w.append(fixed_pos)
        self._fixed_quat_w.append(fixed_quat)

    def _register_terrain(self, scene: Any) -> None:
        if self.backend == "isaaclab":
            mesh = self._isaac_terrain_trimesh(scene)
            self.trimeshes.append(mesh)
            self.meshes_wp.append(scene.ground_mesh)
        else:
            from simple_raycaster.helpers import trimesh2wp

            mesh = self._terrain_trimesh(scene)
            self.trimeshes.append(mesh)
            self.meshes_wp.append(trimesh2wp(mesh, self.device))
        self._append_group(entity=None, body_indices=None)

    def _register_entity(
        self,
        scene: Any,
        entity_name: str,
        *,
        body_pattern: str | None = None,
    ) -> None:
        from simple_raycaster.helpers import trimesh2wp

        entity = scene.entities[entity_name]
        body_indices, body_names, meshes = scene.get_visual_meshes(entity_name)
        if not meshes:
            raise ValueError(
                f"Entity {entity_name!r} has no non-empty visual meshes for warp rendering"
            )
        if not (len(body_indices) == len(body_names) == len(meshes)):
            raise ValueError(
                f"get_visual_meshes({entity_name!r}) returned mismatched lengths: "
                f"indices={len(body_indices)}, names={len(body_names)}, "
                f"meshes={len(meshes)}"
            )

        matcher = re.compile(body_pattern).fullmatch if body_pattern else None
        kept_indices: list[int] = []
        for body_i, body_name, mesh in zip(body_indices, body_names, meshes):
            if matcher is not None and matcher(body_name) is None:
                continue
            if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                warnings.warn(
                    f"MeshRegistry: skipping empty visual mesh for {entity_name!r} "
                    f"body {body_name!r}.",
                    stacklevel=2,
                )
                continue
            self.trimeshes.append(mesh)
            self.meshes_wp.append(trimesh2wp(mesh, self.device))
            kept_indices.append(int(body_i))
        if not kept_indices:
            detail = (
                f" matching body pattern {body_pattern!r}"
                if body_pattern is not None
                else ""
            )
            raise ValueError(
                f"Entity {entity_name!r} has no non-empty visual meshes{detail} "
                f"for warp rendering (available bodies: {body_names})"
            )
        self._append_group(entity=entity, body_indices=tuple(kept_indices))

    def _register_extra(self, scene: Any, target: str) -> None:
        """Register a static Isaac ``AssetBaseCfg`` extra (collision-only prop).

        Geometry and world poses are captured once at registration — these prims
        do not move during the episode.
        """
        if self.backend != "isaaclab":
            raise KeyError(
                f"Static extra target {target!r} requires isaaclab "
                f"(AssetBaseCfg / scene.extras); backend={self.backend!r}"
            )
        from simple_raycaster.helpers import trimesh2wp

        from active_adaptation.envs.backends.isaaclab.meshes import load_prim_trimesh

        view = scene.extras[target]
        prim_paths = getattr(view, "prim_paths", None)
        if not prim_paths:
            raise ValueError(f"Extra {target!r} has no prim_paths for mesh extraction")

        mesh = load_prim_trimesh(prim_paths[0], require_all=True)
        self.trimeshes.append(mesh)
        self.meshes_wp.append(trimesh2wp(mesh, self.device))

        # TODO(known-issue): Isaac Lab 2.3.2 XformPrimView.get_world_poses()
        # often returns only env_0 ([1, 3] / [1, 4]) instead of all instances.
        # Workaround: broadcast that pose across envs via env_origins offsets.
        # Better fix: use a prim view that resolves every ENV_REGEX_NS instance
        # (or wait for an Isaac Lab fix) and call get_world_poses() once with
        # shape (num_envs, …). Safe only while extras stay static.
        pos_w, quat_w = view.get_world_poses()
        pos_w = (pos_w - scene.env_origins[0] + scene.env_origins).reshape(-1, 1, 3)
        quat_w = quat_w.expand(pos_w.shape[0], 1, 4)

        # (num_envs, 1, 3/4) — one pose for the combined extra mesh.
        fixed_pos = pos_w.to(device=self.torch_device).contiguous()
        fixed_quat = quat_w.to(device=self.torch_device).contiguous()
        self._append_group(
            entity=None,
            body_indices=None,
            fixed_pos=fixed_pos,
            fixed_quat=fixed_quat,
        )

    @staticmethod
    def _isaac_terrain_trimesh(scene: Any) -> trimesh.Trimesh:
        """CPU ground trimesh matching :meth:`IsaacSceneAdapter.ground_mesh`."""
        import numpy as np
        from isaaclab.terrains.trimesh.utils import make_plane
        from pxr import UsdGeom
        import isaaclab.sim as sim_utils

        mesh_prim_path = "/World/ground"
        plane_prim = sim_utils.get_first_matching_child_prim(
            mesh_prim_path, lambda prim: prim.GetTypeName() == "Plane"
        )
        if plane_prim is not None:
            return make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)

        mesh_prim = sim_utils.get_first_matching_child_prim(
            mesh_prim_path, lambda prim: prim.GetTypeName() == "Mesh"
        )
        if mesh_prim is None or not mesh_prim.IsValid():
            raise RuntimeError(f"Invalid mesh prim path: {mesh_prim_path}")
        geom = UsdGeom.Mesh(mesh_prim)
        points = np.asarray(geom.GetPointsAttr().Get())
        indices = np.asarray(geom.GetFaceVertexIndicesAttr().Get())
        counts = geom.GetFaceVertexCountsAttr().Get()
        if counts is not None and not np.all(np.asarray(counts) == 3):
            # Fan-triangulate if needed; Isaac ground meshes are usually tris.
            from simple_raycaster.utils_usd import get_trimesh_from_prim

            return get_trimesh_from_prim(mesh_prim)
        faces = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
        return trimesh.Trimesh(vertices=points, faces=faces, process=False)

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
