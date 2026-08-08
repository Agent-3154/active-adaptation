"""Export loaded Gaussians to Viser ``add_gaussian_splats`` arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import numpy as np
import viser
from viser import transforms as tf

if TYPE_CHECKING:
    from fvdb import GaussianSplat3d

# Inria SH DC coefficient (same as aa_scenes.viz.viser / gsplat).
_SH_C0 = 0.28209479177387814


def gaussian_splat_to_viser_arrays(gs: "GaussianSplat3d") -> dict[str, np.ndarray]:
    """Convert an fvdb ``GaussianSplat3d`` to kwargs for ``scene.add_gaussian_splats``.

    Returns:
        Dict with ``centers`` (N,3), ``rgbs`` (N,3), ``opacities`` (N,1),
        ``covariances`` (N,3,3) as float32 numpy arrays on CPU.
    """
    means = gs.means.detach().float().cpu().numpy()
    scales = gs.scales.detach().float().cpu().numpy()
    wxyzs = gs.quats.detach().float().cpu().numpy()
    opacities = gs.opacities.detach().float().cpu().numpy().reshape(-1, 1)
    sh0 = gs.sh0.detach().float().cpu().numpy()[:, 0, :]  # (N, 3)
    rgbs = np.clip(0.5 + _SH_C0 * sh0, 0.0, 1.0).astype(np.float32)

    # R @ diag(s^2) @ R^T  (viser expects second-moment covariances)
    try:
        rs = tf.SO3(wxyzs).as_matrix().astype(np.float32)
    except ImportError:
        # Fallback without viser: build rotation matrices from WXYZ.
        rs = _quat_wxyz_to_matrices(wxyzs)

    scale_sq = scales.astype(np.float32) ** 2
    # einsum: R @ diag(s^2) @ R^T
    covariances = np.einsum(
        "nij,njk,nlk->nil",
        rs,
        np.eye(3, dtype=np.float32)[None, :, :] * scale_sq[:, None, :],
        rs,
    ).astype(np.float32)

    return {
        "centers": means.astype(np.float32),
        "rgbs": rgbs,
        "opacities": opacities.astype(np.float32),
        "covariances": covariances,
    }


def _quat_wxyz_to_matrices(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = wxyz[:, 0], wxyz[:, 1], wxyz[:, 2], wxyz[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r = np.empty((wxyz.shape[0], 3, 3), dtype=np.float32)
    r[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    r[:, 0, 1] = 2.0 * (xy - wz)
    r[:, 0, 2] = 2.0 * (xz + wy)
    r[:, 1, 0] = 2.0 * (xy + wz)
    r[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    r[:, 1, 2] = 2.0 * (yz - wx)
    r[:, 2, 0] = 2.0 * (xz - wy)
    r[:, 2, 1] = 2.0 * (yz + wx)
    r[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return r


def add_gaussian_splat_to_viser_server(
    server: viser.Server,
    gs: "GaussianSplat3d",
    *,
    name: str = "/visual/gaussians",
) -> Any:
    """Upload ``gs`` onto a Viser server scene. Returns the splat handle."""
    arrays = gaussian_splat_to_viser_arrays(gs)
    handle = server.scene.add_gaussian_splats(name, **arrays)
    handle.visible = False
    return handle
