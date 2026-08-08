"""Backend-agnostic visual world (appearance separate from physics)."""

from __future__ import annotations

from typing import Protocol

import torch


class VisualWorld(Protocol):
    """Photoreal / neural appearance for MDP cameras (option A: obs calls render).

    Physics collision meshes stay on :class:`~active_adaptation.envs.adapters.SceneAdapter`.
    Native Kit / mjlab cameras stay on :meth:`SimAdapter.render_sensors`. This
    protocol is for decoupled renderers such as 3DGS (fvdb).
    """

    def load(self) -> None:
        """Load assets onto the device (idempotent)."""
        ...

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
        """Rasterize RGB from world-frame camera poses.

        Args:
            pos_w: ``(N, 3)`` camera origin in world frame.
            quat_wxyz: ``(N, 4)`` camera orientation (WXYZ). Optical frame is
                OpenCV: +Z forward, +Y down, +X right (same as fvdb).
            width / height: image size in pixels.
            fov_y_deg: vertical field of view in degrees.
            near / far: clip planes.

        Returns:
            Float RGB ``(N, H, W, 3)`` in ``[0, 1]``.
        """
        ...
