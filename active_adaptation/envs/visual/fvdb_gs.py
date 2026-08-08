"""fvdb-core Gaussian splat visual world."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import trimesh

from active_adaptation.envs.visual.collision import (
    COLLISION_SUFFIX,
    load_collision_trimesh,
    resolve_collision_usd,
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
    r_wc = matrix_from_quat(quat_wxyz)  # camera axes in world
    # w2c: x_c = R^T (x_w - t)
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
    ) -> None:
        self.ply_path = Path(ply_path) if ply_path is not None else PLY_PATH_PLACEHOLDER
        self.collision_usd = Path(collision_usd) if collision_usd is not None else None
        self.load_collision = bool(load_collision)
        self.device = torch.device(device)
        self.sh_degree_to_use = int(sh_degree_to_use)
        self.min_radius_2d = float(min_radius_2d)
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
                try:
                    self._collision_mesh = load_collision_trimesh(usd)
                    print(
                        f"[FvdbGaussianWorld] Loaded collision mesh from {usd.name}: "
                        f"{len(self._collision_mesh.vertices)} verts, "
                        f"{len(self._collision_mesh.faces)} faces"
                    )
                except Exception as exc:
                    print(f"[FvdbGaussianWorld] Failed to load collision USD {usd}: {exc}")
                    self._collision_mesh = None

    @property
    def gaussian_splat(self):
        """Loaded ``fvdb.GaussianSplat3d`` (after :meth:`load`), or ``None``."""
        return self._gs

    @property
    def collision_mesh(self) -> trimesh.Trimesh | None:
        """World-frame collision trimesh (same frame as the PLY), if loaded."""
        return self._collision_mesh

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
    ) -> torch.Tensor:
        n = pos_w.shape[0]
        w2c = _pos_quat_to_w2c(pos_w, quat_wxyz)
        proj = _pinhole_projection(
            width,
            height,
            fov_y_deg,
            device=pos_w.device,
            dtype=pos_w.dtype,
            batch=n,
        )
        images, _alphas = self._gs.render_images(
            w2c,
            proj,
            width,
            height,
            near=near,
            far=far,
            sh_degree_to_use=self.sh_degree_to_use,
            min_radius_2d=self.min_radius_2d,
        )
        return images
