"""Object grasp-candidate behavior.

Attach via ``AssetSpec(behaviors=(GraspPose(...),))``. Lookup with
``env.require_behavior("chair.grasp")`` when the YAML object key is ``chair``.

Holds a fixed list of **prescribed** object-frame poses ``(pos[3], quat_wxyz[4])``.
No procedural part sampling — callers (or :meth:`for_legs` /
:meth:`for_grasp_board`) decide feasible poses for the whole object.

**Approach convention:** unless an asset or call site says otherwise, the EEF
approaches along its body **+X** (forward). Prescribed grasp quats must align
that axis with the world approach direction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import torch
from typing_extensions import override

from active_adaptation.envs.behaviors.behavior import EntityBehavior
from active_adaptation.utils.math import normalize, quat_from_matrix, quat_rotate

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase

# EEF body-frame forward / approach axis (default unless otherwise specified).
EEF_FORWARD_B: tuple[float, float, float] = (1.0, 0.0, 0.0)

# ``dummy_grasp_board``: 3 heights × 4 orientations per face.
GRASP_BOARD_BARS_PER_FACE: int = 12
_DEFAULT_GRASP_BOARD_PANEL: tuple[float, float, float] = (1.1, 0.04, 1.0)


def grasp_board_bar_specs(
    panel_size: Sequence[float] = _DEFAULT_GRASP_BOARD_PANEL,
) -> list[tuple[str, float, float, tuple[float, float, float]]]:
    """Shared board layout: ``(tag, x, z, axis_xyz)`` per bar (one face).

    Height-major order at bottom / mid / top: horizontal, vertical, −45°, +45°.
    Four X columns keep bars from overlapping. Keep in sync with
    ``build_grasp_board_spec`` / :meth:`GraspPose.for_grasp_board`.
    """
    import math

    width, _thickness, height = (float(x) for x in panel_size)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    z_levels = (
        ("bot", 0.20 * height),
        ("mid", 0.50 * height),
        ("top", 0.80 * height),
    )
    # Columns: h, v, m45 (−45°), p45 (+45°)
    x_cols = (
        -0.34 * width,
        -0.12 * width,
        0.12 * width,
        0.34 * width,
    )
    orients: tuple[tuple[str, tuple[float, float, float]], ...] = (
        ("h", (1.0, 0.0, 0.0)),
        ("v", (0.0, 0.0, 1.0)),
        ("m45", (inv_sqrt2, 0.0, -inv_sqrt2)),
        ("p45", (inv_sqrt2, 0.0, inv_sqrt2)),
    )
    bars: list[tuple[str, float, float, tuple[float, float, float]]] = []
    for level, z in z_levels:
        for (tag_o, axis), x in zip(orients, x_cols, strict=True):
            bars.append((f"{level}_{tag_o}", float(x), float(z), axis))
    return bars


def eef_forward_w(quat_w: torch.Tensor) -> torch.Tensor:
    """World-frame EEF +X (forward / approach) from ``quat_wxyz`` ``[..., 4]``."""
    axis = torch.tensor(
        EEF_FORWARD_B, device=quat_w.device, dtype=quat_w.dtype
    ).expand(quat_w.shape[:-1] + (3,))
    return normalize(quat_rotate(quat_w, axis))


def _frame_x_approach_z_up(
    approach: torch.Tensor, up_hint: torch.Tensor
) -> torch.Tensor:
    """``[N, 3, 3]`` with columns = body x,y,z.

    Default EEF convention (``EEF_FORWARD_B``; fingers open along ±Y):

    - **x** = approach / forward (toward the grasp target)
    - **z** = up_hint projected orthogonal to x
    - **y** = z × x (closing / left-right)
    """
    x = normalize(approach)
    z = up_hint - (up_hint * x).sum(dim=-1, keepdim=True) * x
    z_norm = torch.linalg.norm(z, dim=-1, keepdim=True)
    fallback = torch.zeros_like(up_hint)
    fallback[..., 0] = 1.0
    parallel = z_norm.squeeze(-1) < 1e-6
    if parallel.any():
        z2 = fallback - (fallback * x).sum(dim=-1, keepdim=True) * x
        z = torch.where(parallel.unsqueeze(-1), z2, z)
        z_norm = torch.linalg.norm(z, dim=-1, keepdim=True).clamp_min(1e-8)
    z = z / z_norm
    y = torch.linalg.cross(z, x, dim=-1)
    y = normalize(y)
    # Re-orthogonalize z in case of numeric drift.
    z = torch.linalg.cross(x, y, dim=-1)
    return torch.stack((x, y, z), dim=-1)


class GraspPose(EntityBehavior):
    """Prescribed object-frame grasp poses.

    Args:
        poses: ``[K, 7]`` or sequence of length-7 ``(pos, quat_wxyz)`` in the
            object frame. Quat orients the EEF so **+X** is the approach /
            forward axis (see ``EEF_FORWARD_B``), unless otherwise specified.
            Piper fingers open along **±Y**, so vertical-leg grasps put world
            up on **+Z**.
    """

    name = "grasp"

    def __init__(
        self,
        poses: torch.Tensor | Sequence[Sequence[float]] = (),
    ) -> None:
        super().__init__()
        if isinstance(poses, torch.Tensor):
            poses_t = poses.detach().float().reshape(-1, 7)
        else:
            poses_t = torch.tensor(list(poses), dtype=torch.float32).reshape(-1, 7)
        self._poses_cpu = poses_t.clone()
        self.poses: torch.Tensor | None = None  # set on device in _initialize

    @classmethod
    def for_legs(
        cls,
        corners: Sequence[tuple[float, float]],
        leg_length: float,
        leg_radius: float,
        *,
        z0: float = 0.0,
    ) -> "GraspPose":
        """One prescribed pose per vertical leg: midpoint, looking at center axis.

        Grasp point sits on the **leg axis** (so ±Y jaws can straddle the
        capsule). Approach is horizontal toward ``(0, 0)``. EEF **+X** =
        approach, **+Z** = up the leg, **±Y** = finger closing.
        ``leg_radius`` is kept for callers / future surface offsets.
        """
        del leg_radius  # axis grasp; radius reserved for surface variants
        mid_z = float(z0) + 0.5 * float(leg_length)
        rows: list[list[float]] = []
        for cx, cy in corners:
            outward_xy = torch.tensor([float(cx), float(cy), 0.0])
            n = torch.linalg.norm(outward_xy).clamp_min(1e-8)
            outward = outward_xy / n
            pos = torch.tensor([float(cx), float(cy), mid_z])
            approach = -outward
            up = torch.tensor([0.0, 0.0, 1.0])
            rot = _frame_x_approach_z_up(
                approach.unsqueeze(0), up.unsqueeze(0)
            )[0]
            quat = quat_from_matrix(rot.unsqueeze(0))[0]
            rows.append([*pos.tolist(), *quat.tolist()])
        return cls(poses=rows)

    @classmethod
    def for_door_handles(
        cls,
        door_thickness: float,
        handle_radius: float,
        *,
        handle_shape: str = "capsule",
        handle_box_size: Sequence[float] | None = None,
    ) -> "GraspPose":
        """One prescribed pose per handle face (**handle body frame**).

        Grasp points sit on the handle axis so ±Y jaws can straddle the bar.
        Approach is along ±Y toward the handle origin; EEF **+X** = approach,
        **+Z** = handle up. Callers should transform with the ``handle`` body
        pose (not the articulation root) when the door moves.

        ``handle_shape`` / ``handle_box_size`` match ``dummy_door`` /
        ``dummy_drawer`` (capsule radius vs box half-depth along Y).
        """
        shape = str(handle_shape).lower()
        if shape == "box":
            if handle_box_size is None:
                y_half = max(float(handle_radius), 0.018)
            else:
                depth, _height = handle_box_size[0], handle_box_size[1]
                y_half = 0.5 * float(depth)
        else:
            y_half = float(handle_radius)
        y_off = 0.5 * float(door_thickness) + y_half
        up = torch.tensor([0.0, 0.0, 1.0])
        rows: list[list[float]] = []
        for y_sign, approach_y in ((+1.0, -1.0), (-1.0, +1.0)):
            pos = torch.tensor([0.0, y_sign * y_off, 0.0])
            approach = torch.tensor([0.0, approach_y, 0.0])
            rot = _frame_x_approach_z_up(
                approach.unsqueeze(0), up.unsqueeze(0)
            )[0]
            quat = quat_from_matrix(rot.unsqueeze(0))[0]
            rows.append([*pos.tolist(), *quat.tolist()])
        return cls(poses=rows)

    @classmethod
    def for_drawer_handle(
        cls,
        handle_radius: float = 0.022,
        *,
        handle_shape: str = "capsule",
        handle_box_size: Sequence[float] | None = None,
    ) -> "GraspPose":
        """Front handle grasp in the **handle body frame**.

        Bar lies on ±X at the handle origin (with optional stand-off already in
        the body pose). Approach is from **+Y** toward −Y; EEF **+X** =
        approach, **+Z** = up. Transform with the ``handle_{i}`` body pose.
        """
        del handle_radius, handle_shape, handle_box_size  # axis grasp at origin
        up = torch.tensor([0.0, 0.0, 1.0])
        pos = torch.tensor([0.0, 0.0, 0.0])
        approach = torch.tensor([0.0, -1.0, 0.0])
        rot = _frame_x_approach_z_up(
            approach.unsqueeze(0), up.unsqueeze(0)
        )[0]
        quat = quat_from_matrix(rot.unsqueeze(0))[0]
        return cls(poses=[[*pos.tolist(), *quat.tolist()]])

    @classmethod
    def for_side_axis(
        cls,
        *,
        axis: Sequence[float] = (1.0, 0.0, 0.0),
        approach_dirs: Sequence[Sequence[float]] = (
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
        ),
        pos: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> "GraspPose":
        """Side grasps for a slender body (e.g. horizontal capsule).

        ``axis`` is the long body axis (used as the EEF up-hint so ±Y fingers
        close across the thin cross-section). ``approach_dirs`` are object-frame
        approach directions (normalized). Grasp ``pos`` defaults to the origin.
        """
        axis_t = torch.tensor(list(axis), dtype=torch.float32)
        pos_t = torch.tensor(list(pos), dtype=torch.float32)
        rows: list[list[float]] = []
        for d in approach_dirs:
            approach = torch.tensor(list(d), dtype=torch.float32)
            rot = _frame_x_approach_z_up(
                approach.unsqueeze(0), axis_t.unsqueeze(0)
            )[0]
            quat = quat_from_matrix(rot.unsqueeze(0))[0]
            rows.append([*pos_t.tolist(), *quat.tolist()])
        return cls(poses=rows)

    @classmethod
    def for_grasp_board(
        cls,
        panel_size: Sequence[float] = _DEFAULT_GRASP_BOARD_PANEL,
        handle_length: float = 0.16,
        handle_radius: float = 0.022,
        handle_box_size: Sequence[float] | None = (0.04, 0.03),
        standoff: float = 0.01,
        grasp_clearance: float = 0.02,
    ) -> "GraspPose":
        """Prescribed mid-bar grasps for ``dummy_grasp_board`` (object frame).

        Layout matches ``build_grasp_board_spec`` via :func:`grasp_board_bar_specs`
        (3 heights × hori / vert / −45° / +45° = 12 bars per face). Pose order:

        - indices ``0..11``: **+Y** box face (approach −Y)
        - indices ``12..23``: **−Y** capsule face (approach +Y)

        Grasp points sit on each bar's face-normal line, shifted
        ``grasp_clearance`` outward from the bar center so the gripper need
        not penetrate the bar/panel. EEF **+X** = face approach; long bar
        axis is the up-hint (fingers close across the thin cross-section).
        """
        thickness = float(panel_size[1])
        hr = float(handle_radius)
        so = float(standoff)
        clearance = float(grasp_clearance)
        if clearance < 0.0:
            raise ValueError(f"grasp_clearance must be >= 0, got {grasp_clearance}")
        half_t = 0.5 * thickness
        del handle_length  # axis grasp at bar centers

        if handle_box_size is None:
            hy = max(hr, 0.018)
        else:
            hy = 0.5 * float(handle_box_size[0])

        bars = grasp_board_bar_specs(panel_size)
        rows: list[list[float]] = []
        for y_sign, y_half in ((+1.0, hy), (-1.0, hr)):
            # Bar center + outward clearance (away from panel).
            y = y_sign * (half_t + so + y_half + clearance)
            approach = torch.tensor([0.0, -y_sign, 0.0], dtype=torch.float32)
            for _tag, px, pz, axis in bars:
                pos = torch.tensor([px, y, pz], dtype=torch.float32)
                axis_t = torch.tensor(list(axis), dtype=torch.float32)
                rot = _frame_x_approach_z_up(
                    approach.unsqueeze(0), axis_t.unsqueeze(0)
                )[0]
                quat = quat_from_matrix(rot.unsqueeze(0))[0]
                rows.append([*pos.tolist(), *quat.tolist()])
        return cls(poses=rows)

    @property
    def num_poses(self) -> int:
        if self.poses is not None:
            return int(self.poses.shape[0])
        return int(self._poses_cpu.shape[0])

    @override
    def _initialize(
        self,
        env: "_EnvBase",
        *,
        asset: Any | None = None,
        robot: Any | None = None,
    ) -> None:
        super()._initialize(env, asset=asset, robot=robot)
        if self._poses_cpu.numel() == 0:
            raise ValueError(
                "GraspPose has no poses; pass prescribed poses or use for_legs()"
            )
        self.poses = self._poses_cpu.to(device=self.device)

    def sample_grasp_pose(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample ``n`` poses from the prescribed set. Shape ``[n, 7]``."""
        if not self._initialized or self.poses is None:
            raise RuntimeError("GraspPose is not initialized")
        k = self.poses.shape[0]
        if indices is None:
            indices = torch.randint(
                0, k, (n,), device=self.device, generator=generator
            )
        else:
            indices = indices.to(device=self.device, dtype=torch.long).reshape(-1)
            if indices.numel() != n:
                raise ValueError(
                    f"indices length {indices.numel()} != n={n}"
                )
        return self.poses[indices]

    def sample_grasp_point(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample ``n`` grasp positions. Shape ``[n, 3]``."""
        return self.sample_grasp_pose(n, generator=generator, indices=indices)[
            :, :3
        ]


__all__ = [
    "EEF_FORWARD_B",
    "GRASP_BOARD_BARS_PER_FACE",
    "eef_forward_w",
    "grasp_board_bar_specs",
    "GraspPose",
]
