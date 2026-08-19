"""Visual world factories for active-adaptation envs."""

from __future__ import annotations

from typing import Any

import torch

from .base import VisualWorld

__all__ = [
    "VisualWorld",
    "FvdbGaussianWorld",
    "PLY_PATH_PLACEHOLDER",
    "make_visual_world",
]


def __getattr__(name: str):
    # fvdb_gs / simple_raycaster are optional; A2LocoFlat and other non-GS
    # tasks must not import them at package load.
    if name in ("FvdbGaussianWorld", "PLY_PATH_PLACEHOLDER"):
        from .fvdb_gs import FvdbGaussianWorld, PLY_PATH_PLACEHOLDER

        globals()["FvdbGaussianWorld"] = FvdbGaussianWorld
        globals()["PLY_PATH_PLACEHOLDER"] = PLY_PATH_PLACEHOLDER
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def make_visual_world(
    cfg: Any | None,
    *,
    device: str | torch.device = "cuda",
) -> VisualWorld | None:
    """Build a visual world from Hydra ``task.visual`` (or return ``None``).

    Example YAML::

        visual:
          _target_: fvdb_gs
          ply_path: null   # → PLY_PATH_PLACEHOLDER
          sh_degree_to_use: -1
          min_radius_2d: 0.0
          load_collision: true   # sibling *_collision.usd → Viser mesh
          mesh_entities: [robot] # body visuals → mesh overlay
          composite_meshes: true
          mesh_renderer: diffrast # or raycast (simple_raycaster.mesh_rgbd)
          face_keep: 0.1         # quadric decimate (cache once)
          mesh_chunk_envs: 16    # peak mesh mem ~ O(chunk · V)
          # collision_usd: null  # optional override path
    """
    if cfg is None:
        return None

    if hasattr(cfg, "items"):
        cfg = {k: cfg[k] for k in cfg}
    else:
        cfg = dict(cfg)
    target = str(cfg.pop("_target_", "fvdb_gs"))
    if target in ("fvdb_gs", "FvdbGaussianWorld"):
        from .fvdb_gs import FvdbGaussianWorld

        ply = cfg.pop("ply_path", None)
        return FvdbGaussianWorld(ply_path=ply, device=device, **cfg)
    raise ValueError(f"Unknown visual world target: {target!r}")
