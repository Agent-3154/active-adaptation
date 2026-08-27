"""fvdb-core Gaussian splat visual world (+ optional mesh composite).

Mesh RGB-D is rendered by ``simple_raycaster`` (``diffrast`` / ``raycast``);
this module only depth-composites that RGB-D over the Gaussian image. None of
these replace the MDP depth obs ``extero.raycast_camera``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch
import trimesh
from simple_raycaster import MeshRGBDRenderer

from active_adaptation.envs.visual.collision import (
    COLLISION_SUFFIX,
    load_collision_trimesh,
    resolve_collision_usd,
)
from active_adaptation.envs.visual.mesh_composite import (
    MeshCompositor,
    composite_rgbd,
    make_mesh_compositor,
)
from active_adaptation.utils.math import matrix_from_quat

# TODO: replace with the scene PLY you want for training / play.
# A local pruned InteriorGS file works for smoke tests.
PLY_PATH_PLACEHOLDER = Path(
    # "/home/btx0424/lab51/aa-scenes/data/0002_839955/3dgs_pruned.ply"
    "/home/btx0424/lab51/aa-scenes/data/0001_839920/3dgs_pruned.ply"
)


def _pinhole_projection(
    width: int,
    height: int,
    fov_y_deg: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch: int,
) -> torch.Tensor:
    """Batched ``(N, 3, 3)`` pinhole K matching ``aa_scenes.cameras.pinhole_intrinsics``."""
    fov_y = math.radians(fov_y_deg)
    fy = height / (2.0 * math.tan(fov_y / 2.0))
    fx = fy
    cx = width * 0.5
    cy = height * 0.5
    k = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    return k.unsqueeze(0).expand(batch, -1, -1).contiguous()


def _pos_quat_to_w2c(pos_w: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    """OpenCV camera pose (world) → world-to-camera ``(N, 4, 4)``."""
    r_wc = matrix_from_quat(quat_wxyz)
    r_cw = r_wc.transpose(-1, -2)
    t = -(r_cw @ pos_w.unsqueeze(-1)).squeeze(-1)
    n = pos_w.shape[0]
    w2c = torch.eye(4, device=pos_w.device, dtype=pos_w.dtype).unsqueeze(0).repeat(n, 1, 1)
    w2c[:, :3, :3] = r_cw
    w2c[:, :3, 3] = t
    return w2c


class FvdbGaussianWorld:
    """3DGS appearance via ``fvdb.GaussianSplat3d`` (explicit obs → :meth:`render`).

    InteriorGS scenes also ship ``{id}_collision.usd`` beside the PLY (shared
    Z-up frame). That mesh is loaded for cheap Viser debug; physics collision
    (replacing the ground plane) is not wired yet.

    When entity visual meshes are attached (:meth:`attach_scene_meshes`), each
    :meth:`render` depth-composites mesh RGB-D from the configured
    :class:`~active_adaptation.envs.visual.mesh_composite.MeshCompositor`
    over the Gaussian image. Large batches are processed in
    ``mesh_chunk_envs`` slices to bound peak memory.
    """

    def __init__(
        self,
        ply_path: str | Path | None = None,
        *,
        device: str | torch.device = "cuda",
        sh_degree_to_use: int = -1,
        min_radius_2d: float = 0.0,
        collision_usd: str | Path | None = None,
        load_collision: bool = True,
        mesh_entities: Sequence[str] | None = None,
        composite_meshes: bool = True,
        mesh_renderer: str | MeshCompositor | MeshRGBDRenderer = "raycast",
        face_keep: float = 0.2,
        mesh_chunk_envs: int = 16,
        light_dir: tuple[float, float, float] = (0.45, -0.35, 0.82),
    ) -> None:
        self.ply_path = Path(ply_path) if ply_path is not None else PLY_PATH_PLACEHOLDER
        self.collision_usd = Path(collision_usd) if collision_usd is not None else None
        self.load_collision = bool(load_collision)
        self.device = torch.device(device)
        self.sh_degree_to_use = int(sh_degree_to_use)
        self.min_radius_2d = float(min_radius_2d)
        self.mesh_entities = list(mesh_entities) if mesh_entities is not None else ["robot"]
        self.composite_meshes = bool(composite_meshes)
        self.face_keep = float(face_keep)
        self.mesh_chunk_envs = max(1, int(mesh_chunk_envs))
        self.light_dir = tuple(float(x) for x in light_dir)

        self._mesh: MeshCompositor | None = None
        if self.composite_meshes:
            self._mesh = make_mesh_compositor(
                mesh_renderer,
                device=self.device,
                face_keep=self.face_keep,
                light_dir=self.light_dir,
            )
        self.mesh_renderer = self._mesh.kind if self._mesh is not None else "none"

        self._gs = None
        self._collision_mesh: trimesh.Trimesh | None = None
        self.load()

    def load(self) -> None:
        if self._gs is not None:
            return
        if not self.ply_path.is_file():
            raise FileNotFoundError(
                f"Gaussian splat PLY not found: {self.ply_path}. "
                "Set task.visual.ply_path or update PLY_PATH_PLACEHOLDER."
            )
        from fvdb import GaussianSplat3d

        self._gs, self._meta = GaussianSplat3d.from_ply(str(self.ply_path), device=self.device)

        if self.load_collision:
            usd = self.collision_usd or resolve_collision_usd(self.ply_path)
            if usd is None:
                print(
                    f"[FvdbGaussianWorld] No *{COLLISION_SUFFIX} next to "
                    f"{self.ply_path.parent}; skipping collision mesh viz."
                )
            else:
                self._collision_mesh = load_collision_trimesh(usd)
                print(
                    f"[FvdbGaussianWorld] Loaded collision mesh from {usd.name}: "
                    f"{len(self._collision_mesh.vertices)} verts, "
                    f"{len(self._collision_mesh.faces)} faces"
                )

    @property
    def gaussian_splat(self):
        """Loaded ``fvdb.GaussianSplat3d`` (after :meth:`load`), or ``None``."""
        return self._gs

    @property
    def collision_mesh(self) -> trimesh.Trimesh | None:
        """World-frame collision trimesh (same frame as the PLY), if loaded."""
        return self._collision_mesh

    @property
    def mesh_compositor(self) -> MeshCompositor | None:
        """Active mesh backend, or ``None`` when ``composite_meshes=False``."""
        return self._mesh

    def attach_scene_meshes(self, scene) -> None:
        """Pull visual meshes from ``scene.get_visual_meshes`` for ``mesh_entities``."""
        if self._mesh is None:
            return
        self._mesh.attach_scene_meshes(scene, self.mesh_entities)

    def set_entity_meshes(
        self,
        name: str,
        entity: object,
        meshes: list[trimesh.Trimesh],
    ) -> None:
        """Register / replace one entity's body-local visual meshes."""
        if self._mesh is None:
            raise RuntimeError(
                "composite_meshes=False; cannot register entity meshes "
                "(construct FvdbGaussianWorld with composite_meshes=True)"
            )
        self._mesh.register_entity(name, entity, meshes)

    def _render_gs_rgbd(
        self,
        w2c_cv: torch.Tensor,
        proj_k: torch.Tensor,
        *,
        width: int,
        height: int,
        near: float,
        far: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns RGB ``[N,H,W,3]`` and metric depth ``[N,H,W]`` (far where empty)."""
        out, alphas = self._gs.render_images_and_depths(
            w2c_cv,
            proj_k,
            width,
            height,
            near=near,
            far=far,
            sh_degree_to_use=self.sh_degree_to_use,
            min_radius_2d=self.min_radius_2d,
        )
        rgb = out[..., :-1]
        alpha = alphas[..., 0].clamp_min(1e-6)
        depth = out[..., -1] / alpha
        depth = torch.where(alphas[..., 0] > 1e-3, depth, torch.full_like(depth, far))
        return rgb.clamp(0.0, 1.0), depth

    @torch.no_grad()
    def render(
        self,
        pos_w: torch.Tensor,
        quat_wxyz: torch.Tensor,
        *,
        width: int,
        height: int,
        fov_y_deg: float,
        near: float = 0.05,
        far: float = 50.0,
        origin_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Render GS RGB, optionally depth-composite attached entity meshes.

        Args:
            pos_w / quat_wxyz: OpenCV camera poses in the same frame as the PLY
                (episode-local when ``gs_camera`` uses ``origin: env``).
            origin_w: if set, subtracted from entity ``body_link_pose_w`` so
                meshes match that camera frame (pass ``env.episode_origin``).
        """
        n = pos_w.shape[0]
        w2c_cv = _pos_quat_to_w2c(pos_w, quat_wxyz)
        proj_k = _pinhole_projection(
            width,
            height,
            fov_y_deg,
            device=pos_w.device,
            dtype=pos_w.dtype,
            batch=n,
        )

        mesh = self._mesh
        if mesh is None or not mesh.has_meshes:
            images, _alphas = self._gs.render_images(
                w2c_cv,
                proj_k,
                width,
                height,
                near=near,
                far=far,
                sh_degree_to_use=self.sh_degree_to_use,
                min_radius_2d=self.min_radius_2d,
            )
            return images

        # Chunk over envs so peak mesh tensors stay O(chunk · V), not O(B · V).
        chunk = self.mesh_chunk_envs
        rgb_parts: list[torch.Tensor] = []
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            env_ids = torch.arange(start, end, device=pos_w.device)
            w2c_c = w2c_cv[start:end]
            origin_c = origin_w[start:end] if origin_w is not None else None

            gs_rgb, gs_depth = self._render_gs_rgbd(
                w2c_c,
                proj_k[start:end],
                width=width,
                height=height,
                near=near,
                far=far,
            )
            mesh_rgb, mesh_depth, mesh_mask = mesh.render_rgbd(
                pos_w[start:end],
                quat_wxyz[start:end],
                env_ids=env_ids,
                width=width,
                height=height,
                fov_y_deg=fov_y_deg,
                near=near,
                far=far,
                origin_w=origin_c,
            )
            rgb_parts.append(
                composite_rgbd(gs_rgb, gs_depth, mesh_rgb, mesh_depth, mesh_mask)
            )
        return torch.cat(rgb_parts, dim=0)
