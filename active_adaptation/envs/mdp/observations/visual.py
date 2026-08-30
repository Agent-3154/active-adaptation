"""Observations that read from ``env.visual`` (3DGS / neural appearance)."""

from __future__ import annotations

import math
from typing import Tuple, Literal

import torch
from typing_extensions import override

from active_adaptation.envs.mdp.observations.base import Observation
from active_adaptation.envs.utils import find_bodies
from active_adaptation.utils.math import quat_from_euler_xyz, quat_mul, quat_rotate
from active_adaptation.utils.symmetry import SymmetryTransform

# Mount frame (+X forward, +Y left, +Z up; same as ``raycast_camera``) →
# OpenCV / Viser optical (+Z forward, +Y down, +X right). Same quat as
# ``raycast_camera``'s frustum ``ros_to_cam``.
_MOUNT_TO_OPENCV = (0.5, -0.5, 0.5, -0.5)


class gs_camera(Observation):
    """RGB from ``env.visual.render`` (option A: explicit, not ``render_sensors``).

    Requires ``task.visual`` so ``env.visual`` is a :class:`VisualWorld` (e.g.
    :class:`~active_adaptation.envs.visual.fvdb_gs.FvdbGaussianWorld`).

    Camera pose = body world pose × mount offset. With zero offsets the mount
    matches :class:`~active_adaptation.envs.mdp.observations.extero.raycast_camera`:
    **+X forward, +Y left, +Z up**. ``offset_rpy`` (degrees, XYZ) is applied in
    that mount frame. Internally converted to OpenCV for ``env.visual.render``
    and the Viser frustum.

    Does **not** set ``sensor_render_enabled`` — rasterization happens in
    :meth:`compute`.
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (320, 240),
        fov_y: float = 70.0,
        body_name: str = "torso_link",
        offset_pos: Tuple[float, float, float] = (0.05, 0.0, 0.0),
        offset_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        near: float = 0.05,
        far: float = 50.0,
        origin: Literal["env", "world"] = "env",
    ):
        super().__init__()
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.fov_y = float(fov_y)
        self.body_name = body_name
        self.offset_pos = tuple(float(x) for x in offset_pos)
        self.offset_rpy = tuple(float(x) for x in offset_rpy)
        self.near = float(near)
        self.far = float(far)
        self.origin = origin
        if self.origin not in ["env", "world"]:
            raise ValueError(f"Invalid origin: {self.origin}, must be 'env' or 'world'")
        self._frustum = None

    @override
    def _initialize(self, env):
        super()._initialize(env)
        if self.env.visual is None:
            raise RuntimeError(
                "gs_camera requires env.visual. Add a `visual:` block to the task cfg, e.g.\n"
                "  visual:\n"
                "    _target_: fvdb_gs\n"
                "    ply_path: null  # uses PLY_PATH_PLACEHOLDER\n"
            )
        self.asset = self.env.scene.articulations["robot"]
        body_ids, body_names = find_bodies(self.asset, self.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"gs_camera: expected one body matching {self.body_name!r}, "
                f"got {body_names}"
            )
        self.body_id = int(body_ids[0])
        with torch.device(self.device):
            self._offset_pos = torch.tensor(self.offset_pos, dtype=torch.float32)
            rpy = torch.tensor(self.offset_rpy, dtype=torch.float32) * (math.pi / 180.0)
            self._offset_quat = quat_from_euler_xyz(rpy.unsqueeze(0))
            self._offset_quat = quat_mul(self._offset_quat, torch.tensor(_MOUNT_TO_OPENCV))
            self._image_hwc = torch.zeros(
                self.num_envs, self.resolution[1], self.resolution[0], 3, dtype=torch.float32
            )

        if self.env.sim.has_gui():
            width, height = self.resolution
            try:
                self._viser_viewer = self.env.sim._viser_viewer
                self._frustum = self.env.scene.create_camera_frustum(
                    "gs_camera",
                    fov_y=math.radians(self.fov_y),
                    aspect=width / max(height, 1),
                )
            except Exception as e:
                print(f"Error creating camera frustum: {e}")
                self._viser_viewer = None
                self._frustum = None

    @override
    def update(self) -> None:
        width, height = self.resolution
        body_pos = self.asset.data.body_link_pos_w[:, self.body_id]
        body_quat = self.asset.data.body_link_quat_w[:, self.body_id]
        self.mount_pos_w = body_pos + quat_rotate(
            body_quat, self._offset_pos.expand(self.num_envs, -1)
        )
        self.mount_quat_w = quat_mul(
            body_quat, self._offset_quat.expand(self.num_envs, -1)
        )
        
        # Explicit call — not via sim.render_sensors().
        pos = self.mount_pos_w
        origin_w = None
        if self.origin == "env":
            origin_w = self.env.episode_origin
            pos = pos - origin_w
        self._image_hwc = self.env.visual.render(
            pos,
            self.mount_quat_w,
            width=width,
            height=height,
            fov_y_deg=self.fov_y,
            near=self.near,
            far=self.far,
            origin_w=origin_w,
        )

    @override
    def compute(self) -> torch.Tensor:
        # (N, H, W, 3) → (N, C, H, W)
        return self._image_hwc.permute(0, 3, 1, 2).contiguous()

    @override
    def debug_draw(self) -> None:
        if self._frustum is None:
            return
        env_idx = self._viser_viewer.env_idx
        self._frustum.position = self.mount_pos_w[env_idx].cpu().numpy()
        self._frustum.wxyz = self.mount_quat_w[env_idx].cpu().numpy()
        self._frustum.image = (
            (self._image_hwc[env_idx] * 255.0)
            .clamp(0, 255)
            .byte()
            .cpu()
            .numpy()
        )

    @override
    def symmetry_transform(self) -> SymmetryTransform:
        width = self.resolution[0]
        perm = torch.arange(width - 1, -1, -1, dtype=torch.long)
        return SymmetryTransform(perm, torch.ones(width))
