"""GS ↔ mesh depth composition (mesh RGB-D comes from ``simple_raycaster``).

AA owns scene attachment + overlaying mesh RGB-D onto Gaussian images.
Raster / raycast backends live in ``simple_raycaster.mesh_rgbd``.
"""

from __future__ import annotations

from typing import Sequence

import torch
import trimesh
from simple_raycaster import MeshRGBDRenderer, make_mesh_rgbd_renderer


def composite_rgbd(
    gs_rgb: torch.Tensor,
    gs_depth: torch.Tensor,
    mesh_rgb: torch.Tensor,
    mesh_depth: torch.Tensor,
    mesh_mask: torch.Tensor,
) -> torch.Tensor:
    """Depth-test overlay: mesh wins when closer than GS."""
    use_mesh = mesh_mask & (mesh_depth < gs_depth)
    return torch.where(use_mesh.unsqueeze(-1), mesh_rgb, gs_rgb)


class MeshCompositor:
    """Pull entity visuals from the env scene and depth-composite over GS."""

    def __init__(
        self,
        renderer: str | MeshRGBDRenderer = "diffrast",
        *,
        device: torch.device,
        face_keep: float = 0.2,
        light_dir: tuple[float, float, float] = (0.45, -0.35, 0.82),
    ) -> None:
        self._renderer = make_mesh_rgbd_renderer(
            renderer,
            device=device,
            face_keep=face_keep,
            light_dir=light_dir,
        )
        self.light_dir = tuple(float(x) for x in light_dir)

    @property
    def kind(self) -> str:
        return self._renderer.kind

    @property
    def has_meshes(self) -> bool:
        return self._renderer.has_meshes

    def clear(self) -> None:
        self._renderer.clear()

    def register_entity(
        self,
        name: str,
        entity: object,
        meshes: list[trimesh.Trimesh],
    ) -> None:
        self._renderer.register_entity(name, entity, meshes)

    def attach_scene_meshes(self, scene, entity_names: Sequence[str]) -> None:
        """Pull visuals for ``entity_names`` from ``scene.get_visual_meshes``."""
        self.clear()
        for name in entity_names:
            if name not in scene.entities:
                raise KeyError(
                    f"mesh entity {name!r} not in scene.entities "
                    f"(have {sorted(scene.entities)})"
                )
            meshes = scene.get_visual_meshes(name)
            self.register_entity(name, scene.entities[name], meshes)

    def render_rgbd(
        self,
        cam_pos_w: torch.Tensor,
        cam_quat_wxyz: torch.Tensor,
        *,
        env_ids: torch.Tensor,
        width: int,
        height: int,
        fov_y_deg: float,
        near: float,
        far: float,
        origin_w: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._renderer.render(
            cam_pos_w,
            cam_quat_wxyz,
            env_ids=env_ids,
            origin_w=origin_w,
            width=width,
            height=height,
            fov_y_deg=fov_y_deg,
            near=near,
            far=far,
            light_dir=self.light_dir,
        )


def make_mesh_compositor(
    kind: str | MeshCompositor | MeshRGBDRenderer,
    *,
    device: torch.device,
    face_keep: float = 0.2,
    light_dir: tuple[float, float, float] = (0.45, -0.35, 0.82),
) -> MeshCompositor:
    if isinstance(kind, MeshCompositor):
        return kind
    return MeshCompositor(
        kind,
        device=device,
        face_keep=face_keep,
        light_dir=light_dir,
    )
