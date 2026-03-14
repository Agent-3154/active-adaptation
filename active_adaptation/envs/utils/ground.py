from __future__ import annotations

import numpy as np
import torch
import warp as wp

from active_adaptation.utils.warp import raycast_mesh


def _initialize_isaac_ground_mesh(mesh_prim_path: str, device: str) -> wp.Mesh:
    import isaaclab.sim as sim_utils
    from isaaclab.terrains.trimesh.utils import make_plane
    from isaaclab.utils.warp import convert_to_warp_mesh
    from pxr import UsdGeom

    mesh_prim = sim_utils.get_first_matching_child_prim(
        mesh_prim_path, lambda prim: prim.GetTypeName() == "Plane"
    )
    if mesh_prim is None:
        mesh_prim = sim_utils.get_first_matching_child_prim(
            mesh_prim_path, lambda prim: prim.GetTypeName() == "Mesh"
        )
        if mesh_prim is None or not mesh_prim.IsValid():
            raise RuntimeError(f"Invalid mesh prim path: {mesh_prim_path}")
        mesh_prim = UsdGeom.Mesh(mesh_prim)
        points = np.asarray(mesh_prim.GetPointsAttr().Get())
        indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
        return convert_to_warp_mesh(points, indices, device=device)

    plane_mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
    return convert_to_warp_mesh(plane_mesh.vertices, plane_mesh.faces, device=device)


class GroundQuery:
    def __init__(self, scene, backend: str, terrain_type: str | None, device: torch.device):
        self.scene = scene
        self.backend = backend
        self.terrain_type = terrain_type
        self.device = device
        self._mesh: wp.Mesh | None = None

    @property
    def mesh(self) -> wp.Mesh:
        if self._mesh is None:
            self._mesh = self._build_mesh()
        return self._mesh

    def height_at(self, pos: torch.Tensor) -> torch.Tensor:
        if self.terrain_type == "plane":
            return torch.zeros(pos.shape[:-1], device=pos.device, dtype=pos.dtype)

        bshape = pos.shape[:-1]
        ray_starts = pos.reshape(-1, 3)
        ray_directions = torch.tensor(
            [0.0, 0.0, -1.0], device=ray_starts.device, dtype=ray_starts.dtype
        ).expand(ray_starts.shape[0], 3)
        _, ray_distances = raycast_mesh(
            ray_starts=ray_starts,
            ray_directions=ray_directions,
            min_dist=0.0,
            max_dist=100.0,
            mesh=self.mesh,
        )
        ray_distance = ray_distances.reshape(-1).nan_to_num(posinf=100.0)
        return (ray_starts[:, 2] - ray_distance).reshape(*bshape)

    def _build_mesh(self) -> wp.Mesh:
        if self.backend == "isaac":
            return _initialize_isaac_ground_mesh("/World/ground", self.device.type)
        if self.backend in {"mujoco", "mjlab"}:
            return wp.Mesh(
                points=wp.array(
                    self.scene.ground_mesh.vertices,
                    dtype=wp.vec3,
                    device=self.device.type,
                ),
                indices=wp.array(
                    self.scene.ground_mesh.faces.flatten(),
                    dtype=wp.int32,
                    device=self.device.type,
                ),
            )
        raise NotImplementedError(f"Unsupported backend: {self.backend}")
