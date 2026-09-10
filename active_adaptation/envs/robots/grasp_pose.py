"""Object grasp-candidate adaptation.

Attach via ``AssetSpec(adaptations=(GraspPose(...),))``. Lookup with
``env.require_adaptation("chair.grasp")`` when the YAML object key is ``chair``.

Holds a fixed list of **prescribed** object-frame poses ``(pos[3], quat_wxyz[4])``.
No procedural part sampling — callers (or :meth:`for_legs`) decide feasible
poses for the whole object.

**Approach convention:** unless an asset or call site says otherwise, the EEF
approaches along its body **+X** (forward). Prescribed grasp quats must align
that axis with the world approach direction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import torch
from typing_extensions import override

from active_adaptation.envs.robots.adaptation import RobotAdaptation
from active_adaptation.utils.math import normalize, quat_from_matrix, quat_rotate

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase

# EEF body-frame forward / approach axis (default unless otherwise specified).
EEF_FORWARD_B: tuple[float, float, float] = (1.0, 0.0, 0.0)


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


class GraspPose(RobotAdaptation):
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
    ) -> "GraspPose":
        """One prescribed pose per handle face (**handle body frame**).

        Grasp points sit on the handle axis so ±Y jaws can straddle the
        capsule. Approach is along ±Y toward the handle origin; EEF **+X** =
        approach, **+Z** = handle up. Callers should transform with the
        ``handle`` body pose (not the articulation root) when the door moves.
        """
        y_off = 0.5 * float(door_thickness) + float(handle_radius)
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


__all__ = ["EEF_FORWARD_B", "eef_forward_w", "GraspPose"]
