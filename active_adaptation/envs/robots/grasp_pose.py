"""Object grasp-candidate adaptation.

Attach to a graspable scene object via ``AssetSpec(adaptations=(GraspPose(...),))``.
After bind, look up with ``env.require_adaptation("object.grasp")`` when the
YAML key under ``objects:`` is ``object``.

Composition: ``GraspPose`` holds a list of primitive parts (currently
``CapsuleGrasp`` for furniture legs). Table and chair share the same adaptation;
only the part list differs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import torch
from typing_extensions import override

from active_adaptation.envs.robots.adaptation import RobotAdaptation
from active_adaptation.utils.math import normalize, quat_from_matrix

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


@dataclass(frozen=True)
class CapsuleGrasp:
    """Grasp samples on a capsule in the **object frame**.

    ``fromto`` is ``(x0, y0, z0, x1, y1, z1)`` — the capsule axis endpoints
    (MuJoCo / furniture-leg convention). ``t_range`` is the fractional range
    along that axis used for contact (avoid floor / joint ends).
    """

    fromto: tuple[float, float, float, float, float, float]
    radius: float
    t_range: tuple[float, float] = (0.2, 0.8)
    on_surface: bool = True

    def __post_init__(self) -> None:
        t0, t1 = self.t_range
        if not (0.0 <= t0 < t1 <= 1.0):
            raise ValueError(f"Invalid t_range {self.t_range}; need 0 <= t0 < t1 <= 1")
        if self.radius < 0.0:
            raise ValueError(f"radius must be >= 0, got {self.radius}")

    @property
    def p0(self) -> torch.Tensor:
        return torch.tensor(self.fromto[:3], dtype=torch.float32)

    @property
    def p1(self) -> torch.Tensor:
        return torch.tensor(self.fromto[3:], dtype=torch.float32)

    @property
    def length(self) -> float:
        d = self.p1 - self.p0
        return float(torch.linalg.norm(d).item())


def _frame_from_approach_up(approach: torch.Tensor, up_hint: torch.Tensor) -> torch.Tensor:
    """Build ``[N, 3, 3]`` rotation matrices: columns = body x,y,z in object frame.

    - z = approach (toward the grasp target)
    - y = up_hint projected orthogonal to z (leg axis)
    - x = y × z (closing / lateral)
    """
    z = normalize(approach)
    # Remove component along z so y stays orthogonal.
    y = up_hint - (up_hint * z).sum(dim=-1, keepdim=True) * z
    y_norm = torch.linalg.norm(y, dim=-1, keepdim=True)
    # Degenerate: approach nearly parallel to up — pick a horizontal fallback.
    fallback = torch.zeros_like(up_hint)
    fallback[..., 0] = 1.0
    parallel = y_norm.squeeze(-1) < 1e-6
    if parallel.any():
        y2 = fallback - (fallback * z).sum(dim=-1, keepdim=True) * z
        y = torch.where(parallel.unsqueeze(-1), y2, y)
        y_norm = torch.linalg.norm(y, dim=-1, keepdim=True).clamp_min(1e-8)
    y = y / y_norm
    x = torch.linalg.cross(y, z, dim=-1)
    x = normalize(x)
    # Re-orthogonalize y in case of numeric drift.
    y = torch.linalg.cross(z, x, dim=-1)
    return torch.stack((x, y, z), dim=-1)


class GraspPose(RobotAdaptation):
    """Sample grasp candidates on the bound object (object frame).

    Args:
        parts: Primitive grasp regions. Use :meth:`for_legs` for furniture.
    """

    name = "grasp"

    def __init__(self, parts: Sequence[CapsuleGrasp] = ()) -> None:
        super().__init__()
        self.parts: tuple[CapsuleGrasp, ...] = tuple(parts)
        self._part_lengths: torch.Tensor | None = None
        self._part_p0: torch.Tensor | None = None  # [P, 3]
        self._part_axis: torch.Tensor | None = None  # [P, 3] unit
        self._part_radius: torch.Tensor | None = None  # [P]
        self._part_t0: torch.Tensor | None = None
        self._part_t1: torch.Tensor | None = None
        self._part_on_surface: torch.Tensor | None = None

    @classmethod
    def for_legs(
        cls,
        corners: Sequence[tuple[float, float]],
        leg_length: float,
        leg_radius: float,
        *,
        t_range: tuple[float, float] = (0.2, 0.8),
        z0: float = 0.0,
    ) -> "GraspPose":
        """Build a grasp adaptation for vertical furniture legs."""
        parts = [
            CapsuleGrasp(
                fromto=(float(x), float(y), float(z0), float(x), float(y), float(leg_length)),
                radius=float(leg_radius),
                t_range=t_range,
            )
            for x, y in corners
        ]
        return cls(parts=parts)

    @override
    def _initialize(
        self,
        env: "_EnvBase",
        *,
        asset: Any | None = None,
        robot: Any | None = None,
    ) -> None:
        super()._initialize(env, asset=asset, robot=robot)
        if not self.parts:
            raise ValueError("GraspPose has no parts; pass CapsuleGrasp list or use for_legs()")
        device = self.device
        p0 = torch.stack([p.p0 for p in self.parts], dim=0).to(device)
        p1 = torch.stack([p.p1 for p in self.parts], dim=0).to(device)
        axis = p1 - p0
        lengths = torch.linalg.norm(axis, dim=-1).clamp_min(1e-8)
        self._part_p0 = p0
        self._part_axis = axis / lengths.unsqueeze(-1)
        self._part_lengths = lengths
        self._part_radius = torch.tensor(
            [p.radius for p in self.parts], device=device, dtype=torch.float32
        )
        self._part_t0 = torch.tensor(
            [p.t_range[0] for p in self.parts], device=device, dtype=torch.float32
        )
        self._part_t1 = torch.tensor(
            [p.t_range[1] for p in self.parts], device=device, dtype=torch.float32
        )
        self._part_on_surface = torch.tensor(
            [p.on_surface for p in self.parts], device=device, dtype=torch.bool
        )

    def _sample_part_indices(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Sample part ids proportional to capsule length."""
        assert self._part_lengths is not None
        weights = self._part_lengths / self._part_lengths.sum()
        return torch.multinomial(weights, n, replacement=True, generator=generator)

    def _sample_contacts(
        self, n: int, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(pos[n,3], approach[n,3], up[n,3])`` in the object frame."""
        if not self._initialized:
            raise RuntimeError("GraspPose is not initialized")
        assert self._part_p0 is not None and self._part_axis is not None
        assert self._part_t0 is not None and self._part_t1 is not None
        assert self._part_radius is not None and self._part_on_surface is not None

        idx = self._sample_part_indices(n, generator=generator)
        p0 = self._part_p0[idx]
        axis = self._part_axis[idx]
        length = self._part_lengths[idx]
        t0 = self._part_t0[idx]
        t1 = self._part_t1[idx]
        radius = self._part_radius[idx]
        on_surface = self._part_on_surface[idx]

        u = torch.rand(n, device=self.device, generator=generator)
        t = t0 + (t1 - t0) * u
        axis_point = p0 + axis * (t * length).unsqueeze(-1)

        # Random azimuth in the plane perpendicular to the capsule axis.
        # Build an orthonormal basis (e1, e2) ⊥ axis.
        ref = torch.zeros(n, 3, device=self.device)
        # Prefer world-X unless nearly parallel to axis.
        ref[:, 0] = 1.0
        parallel = (axis * ref).sum(dim=-1).abs() > 0.9
        ref[parallel] = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        e1 = normalize(torch.linalg.cross(axis, ref, dim=-1))
        e2 = torch.linalg.cross(axis, e1, dim=-1)

        theta = torch.rand(n, device=self.device, generator=generator) * (2.0 * torch.pi)
        cos_t = torch.cos(theta).unsqueeze(-1)
        sin_t = torch.sin(theta).unsqueeze(-1)
        radial = e1 * cos_t + e2 * sin_t

        pos = axis_point + torch.where(
            on_surface.unsqueeze(-1),
            radial * radius.unsqueeze(-1),
            torch.zeros_like(radial),
        )
        # Approach toward the axis (side grasp from outside).
        approach = -radial
        up = axis
        return pos, approach, up

    def sample_grasp_point(
        self, n: int, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Sample ``n`` grasp points in the object frame. Shape ``[n, 3]``."""
        pos, _, _ = self._sample_contacts(n, generator=generator)
        return pos

    def sample_grasp_pose(
        self, n: int, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Sample ``n`` grasp poses in the object frame.

        Returns:
            Tensor of shape ``[n, 7]`` as ``(pos[3], quat_wxyz[4])``.
            Gripper frame: **z** = approach toward the leg axis, **y** = along
            the capsule, **x** = y × z.
        """
        pos, approach, up = self._sample_contacts(n, generator=generator)
        rot = _frame_from_approach_up(approach, up)
        quat = quat_from_matrix(rot)
        return torch.cat((pos, quat), dim=-1)


__all__ = ["CapsuleGrasp", "GraspPose"]
