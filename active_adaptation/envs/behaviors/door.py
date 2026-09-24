"""Door articulation behavior: lock, open direction, handle unlock.

Attach via ``AssetSpec(behaviors=(DoorBehavior(), ...))``. Lookup with
``env.require_behavior("door.door")`` when the YAML object key is ``door``.

**Locking:** while locked, the hinge (``door_joint``) is held at ``0`` with high
stiffness (Isaac) and/or near-zero joint limits. Unlock when
``|handle_joint|`` exceeds the threshold (default 30°).

**Open direction** (object frame: hinge +Z, panel +X, through-door +Y):

- ``"pull"`` (+1): allow ``door_joint >= 0`` (panel swings toward **+Y**).
  Slide joint stays locked at 0. Handle is a horizontal lever.
- ``"push"`` (−1): allow ``door_joint <= 0`` (panel swings toward **−Y**).
  Slide joint stays locked at 0. Handle is a horizontal lever.
- ``"slide"`` (0): allow ``door_slide_joint >= 0`` (panel translates along
  **+X**, toward the latch). The hinge stays locked at 0. The handle is held
  at ``+π/2`` (vertical bar) with high stiffness, and the slide is never
  locked — no handle twist is required.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal, Sequence

import torch
from typing_extensions import override

from active_adaptation.envs.behaviors.behavior import EntityBehavior
from active_adaptation.envs.utils import find_joints

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase

OpenDirection = Literal["push", "pull", "slide"]

# Signed open direction in the door frame (+ = pull / +Y, − = push / −Y).
# Slide is unsigned: positive ``door_slide_joint`` opens along +X.
DIR_PULL = 1
DIR_PUSH = -1
DIR_SLIDE = 0
# ``handle_joint`` about +Y. +π/2 stands the horizontal bar up along +Z.
SLIDE_HANDLE_ANGLE = math.pi / 2.0


def _parse_open_direction(direction: OpenDirection | int | str) -> int:
    if isinstance(direction, int):
        if direction not in (DIR_PULL, DIR_PUSH, DIR_SLIDE):
            raise ValueError(
                f"open_direction int must be +1 (pull), -1 (push), or 0 (slide), "
                f"got {direction}"
            )
        return direction
    key = str(direction).lower()
    if key == "pull":
        return DIR_PULL
    if key == "push":
        return DIR_PUSH
    if key == "slide":
        return DIR_SLIDE
    raise ValueError(
        f"open_direction must be 'push', 'pull', or 'slide', got {direction!r}"
    )


class DoorBehavior(EntityBehavior):
    """Control door lock, open direction, and handle-based unlock."""

    name = "door"

    def __init__(
        self,
        door_joint_name: str = "door_joint",
        slide_joint_name: str = "door_slide_joint",
        handle_joint_name: str = "handle_joint",
        *,
        locked_stiffness: float = 500.0,
        unlocked_stiffness: float = 0.0,
        locked_damping: float = 50.0,
        unlocked_damping: float = 4.0,
        slide_handle_stiffness: float = 400.0,
        slide_handle_damping: float = 20.0,
        handle_unlock_threshold_deg: float = 30.0,
        open_direction: OpenDirection = "pull",
        initially_locked: bool = True,
        lock_limit_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.door_joint_name_cfg = door_joint_name
        self.slide_joint_name_cfg = slide_joint_name
        self.handle_joint_name_cfg = handle_joint_name
        self.locked_stiffness = float(locked_stiffness)
        self.unlocked_stiffness = float(unlocked_stiffness)
        self.locked_damping = float(locked_damping)
        self.unlocked_damping = float(unlocked_damping)
        self.slide_handle_stiffness = float(slide_handle_stiffness)
        self.slide_handle_damping = float(slide_handle_damping)
        self.handle_unlock_threshold = math.radians(float(handle_unlock_threshold_deg))
        self.default_open_direction = _parse_open_direction(open_direction)
        self.initially_locked = bool(initially_locked)
        if self.default_open_direction == DIR_SLIDE:
            self.initially_locked = False
        self.lock_limit_eps = float(lock_limit_eps)

        self.door_joint_id: int = -1
        self.slide_joint_id: int = -1
        self.handle_joint_id: int = -1
        self._door_range_lo: float = -math.pi
        self._door_range_hi: float = math.pi
        self._slide_range_lo: float = 0.0
        self._slide_range_hi: float = 0.9

        self.locked: torch.Tensor | None = None  # [N] bool
        self.open_dir: torch.Tensor | None = None  # [N] long ±1
        self._gains_dirty: bool = True

    @override
    def _initialize(
        self,
        env: "_EnvBase",
        *,
        asset: Any | None = None,
        robot: Any | None = None,
    ) -> None:
        super()._initialize(env, asset=asset, robot=robot)

        door_ids, door_names = find_joints(self.asset, self.door_joint_name_cfg)
        if len(door_ids) != 1:
            raise ValueError(
                f"DoorBehavior: expected one door joint "
                f"{self.door_joint_name_cfg!r}, got {door_names}"
            )
        handle_ids, handle_names = find_joints(self.asset, self.handle_joint_name_cfg)
        if len(handle_ids) != 1:
            raise ValueError(
                f"DoorBehavior: expected one handle joint "
                f"{self.handle_joint_name_cfg!r}, got {handle_names}"
            )
        self.door_joint_id = int(door_ids[0])
        self.handle_joint_id = int(handle_ids[0])
        slide_ids, slide_names = find_joints(self.asset, self.slide_joint_name_cfg)
        if len(slide_ids) != 1:
            raise ValueError(
                f"DoorBehavior: expected one slide joint "
                f"{self.slide_joint_name_cfg!r}, got {slide_names}"
            )
        self.slide_joint_id = int(slide_ids[0])

        limits = self.asset.data.joint_pos_limits[0, self.door_joint_id]
        self._door_range_lo = float(limits[0].item())
        self._door_range_hi = float(limits[1].item())
        slide_limits = self.asset.data.joint_pos_limits[0, self.slide_joint_id]
        self._slide_range_lo = float(slide_limits[0].item())
        self._slide_range_hi = float(slide_limits[1].item())

        n = self.num_envs
        self.locked = torch.full(
            (n,), self.initially_locked, device=self.device, dtype=torch.bool
        )
        self.open_dir = torch.full(
            (n,),
            self.default_open_direction,
            device=self.device,
            dtype=torch.long,
        )
        self._gains_dirty = True
        self._apply_lock_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def door_joint_pos(self) -> torch.Tensor:
        return self.asset.data.joint_pos[:, self.door_joint_id]

    @property
    def slide_joint_pos(self) -> torch.Tensor:
        return self.asset.data.joint_pos[:, self.slide_joint_id]

    @property
    def handle_joint_pos(self) -> torch.Tensor:
        return self.asset.data.joint_pos[:, self.handle_joint_id]

    def set_locked(
        self,
        locked: bool | torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Set lock flag (``True`` = hinge held at 0)."""
        assert self.locked is not None
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if isinstance(locked, bool):
            self.locked[env_ids] = locked
        else:
            self.locked[env_ids] = locked.to(device=self.device, dtype=torch.bool)
        self._gains_dirty = True
        self._apply_lock_state(env_ids)

    def set_open_direction(
        self,
        direction: OpenDirection | int | str | Sequence[OpenDirection] | torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Set pull / push / slide per env (see module docstring)."""
        assert self.open_dir is not None
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if isinstance(direction, torch.Tensor):
            vals = direction.to(device=self.device, dtype=torch.long).reshape(-1)
        elif isinstance(direction, (list, tuple)):
            vals = torch.tensor(
                [_parse_open_direction(d) for d in direction],
                device=self.device,
                dtype=torch.long,
            )
        else:
            vals = torch.full(
                (env_ids.numel(),),
                _parse_open_direction(direction),
                device=self.device,
                dtype=torch.long,
            )
        if vals.numel() != env_ids.numel():
            raise ValueError(
                f"open_direction length {vals.numel()} != env_ids {env_ids.numel()}"
            )
        self.open_dir[env_ids] = vals
        slide = vals == DIR_SLIDE
        if slide.any():
            self.locked[env_ids[slide]] = False
        self._gains_dirty = True
        self._apply_lock_state(env_ids)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @override
    def reset(self, env_ids: torch.Tensor, tensordict: Any = None) -> None:
        assert self.locked is not None and self.open_dir is not None
        self.locked[env_ids] = self.initially_locked
        self.open_dir[env_ids] = self.default_open_direction
        self._gains_dirty = True
        self._apply_lock_state(env_ids)

    @override
    def update(self, tensordict: Any = None) -> None:
        """Auto-unlock hinged doors when the handle rotates past the threshold.

        Slide mode is always unlocked; the vertical handle is a stiff grip.
        """
        assert self.locked is not None and self.open_dir is not None
        hinged = self.open_dir != DIR_SLIDE
        unlocked_by_handle = (
            self.handle_joint_pos.abs() >= self.handle_unlock_threshold
        )
        newly = self.locked & hinged & unlocked_by_handle
        if newly.any():
            self.locked = self.locked & ~newly
            self._gains_dirty = True
            self._apply_lock_state(newly.nonzero().squeeze(-1))

    @override
    def pre_step(self, substep: int) -> None:
        """Hold locked hinges and unused slide/hinge DOFs; keep slide handles vertical."""
        del substep
        assert self.locked is not None and self.open_dir is not None
        if self._gains_dirty:
            self._apply_lock_state()
        slide = self.open_dir == DIR_SLIDE
        hold_hinge = self.locked | slide
        if hold_hinge.any():
            ids = hold_hinge.nonzero().squeeze(-1)
            zeros = torch.zeros(ids.numel(), 1, device=self.device)
            self.asset.set_joint_position_target(
                zeros, joint_ids=[self.door_joint_id], env_ids=ids
            )
        hold_slide = ~slide
        if hold_slide.any():
            ids = hold_slide.nonzero().squeeze(-1)
            zeros = torch.zeros(ids.numel(), 1, device=self.device)
            self.asset.set_joint_position_target(
                zeros, joint_ids=[self.slide_joint_id], env_ids=ids
            )
        if slide.any():
            ids = slide.nonzero().squeeze(-1)
            vertical = torch.full(
                (ids.numel(), 1), SLIDE_HANDLE_ANGLE, device=self.device
            )
            self.asset.set_joint_position_target(
                vertical, joint_ids=[self.handle_joint_id], env_ids=ids
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _door_limits_for_state(
        self, locked: torch.Tensor, open_dir: torch.Tensor
    ) -> torch.Tensor:
        """``[M, 1, 2]`` limits for the door joint given per-row state."""
        m = locked.shape[0]
        lo = torch.full((m,), self._door_range_lo, device=self.device)
        hi = torch.full((m,), self._door_range_hi, device=self.device)
        # Unlocked: one-sided swing.
        pull = open_dir > 0
        lo = torch.where(pull, torch.zeros_like(lo), lo)
        hi = torch.where(pull, hi, torch.zeros_like(hi))
        # Locked: clamp near closed.
        eps = self.lock_limit_eps
        lo = torch.where(locked, torch.full_like(lo, -eps), lo)
        hi = torch.where(locked, torch.full_like(hi, eps), hi)
        return torch.stack((lo, hi), dim=-1).unsqueeze(1)  # [M, 1, 2]

    def _apply_lock_state(self, env_ids: torch.Tensor | None = None) -> None:
        assert self.locked is not None and self.open_dir is not None
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if env_ids.numel() == 0:
            return

        locked = self.locked[env_ids]
        open_dir = self.open_dir[env_ids]
        slide = open_dir == DIR_SLIDE
        hinge_locked = locked | slide
        limits = self._door_limits_for_state(hinge_locked, open_dir)
        slide_limits = self._slide_limits_for_state(slide)

        if hasattr(self.asset, "write_joint_position_limit_to_sim"):
            self.asset.write_joint_position_limit_to_sim(
                limits,
                joint_ids=[self.door_joint_id],
                env_ids=env_ids,
                warn_limit_violation=False,
            )
            self.asset.write_joint_position_limit_to_sim(
                slide_limits,
                joint_ids=[self.slide_joint_id],
                env_ids=env_ids,
                warn_limit_violation=False,
            )
        elif self.env.backend == "mjlab":
            self._mjlab_write_jnt_range(
                env_ids, limits.squeeze(1), self.door_joint_id
            )
            self._mjlab_write_jnt_range(
                env_ids, slide_limits.squeeze(1), self.slide_joint_id
            )

        # Stiffness / damping (Isaac ImplicitActuator; best-effort elsewhere).
        stiff = torch.where(
            hinge_locked,
            torch.full((env_ids.numel(),), self.locked_stiffness, device=self.device),
            torch.full((env_ids.numel(),), self.unlocked_stiffness, device=self.device),
        ).unsqueeze(-1)
        damp = torch.where(
            hinge_locked,
            torch.full((env_ids.numel(),), self.locked_damping, device=self.device),
            torch.full((env_ids.numel(),), self.unlocked_damping, device=self.device),
        ).unsqueeze(-1)
        slide_stiff = torch.where(
            slide,
            torch.full((env_ids.numel(),), self.unlocked_stiffness, device=self.device),
            torch.full((env_ids.numel(),), self.locked_stiffness, device=self.device),
        ).unsqueeze(-1)
        slide_damp = torch.where(
            slide,
            torch.full((env_ids.numel(),), self.unlocked_damping, device=self.device),
            torch.full((env_ids.numel(),), self.locked_damping, device=self.device),
        ).unsqueeze(-1)

        if hasattr(self.asset, "write_joint_stiffness_to_sim"):
            self.asset.write_joint_stiffness_to_sim(
                stiff, joint_ids=[self.door_joint_id], env_ids=env_ids
            )
            self.asset.write_joint_damping_to_sim(
                damp, joint_ids=[self.door_joint_id], env_ids=env_ids
            )
            self.asset.write_joint_stiffness_to_sim(
                slide_stiff, joint_ids=[self.slide_joint_id], env_ids=env_ids
            )
            self.asset.write_joint_damping_to_sim(
                slide_damp, joint_ids=[self.slide_joint_id], env_ids=env_ids
            )
            if slide.any():
                ids = env_ids[slide]
                handle_kp = torch.full(
                    (ids.numel(), 1), self.slide_handle_stiffness, device=self.device
                )
                handle_kd = torch.full(
                    (ids.numel(), 1), self.slide_handle_damping, device=self.device
                )
                self.asset.write_joint_stiffness_to_sim(
                    handle_kp, joint_ids=[self.handle_joint_id], env_ids=ids
                )
                self.asset.write_joint_damping_to_sim(
                    handle_kd, joint_ids=[self.handle_joint_id], env_ids=ids
                )
        elif self.env.backend == "mjlab":
            self._mjlab_write_door_pd_gains(
                env_ids, stiff.squeeze(-1), damp.squeeze(-1), "door_joint"
            )
            self._mjlab_write_door_pd_gains(
                env_ids,
                slide_stiff.squeeze(-1),
                slide_damp.squeeze(-1),
                "door_slide_joint",
            )

        self._gains_dirty = False

    def _slide_limits_for_state(self, slide: torch.Tensor) -> torch.Tensor:
        """``[M, 1, 2]`` slide limits. Slide mode uses the full +X travel."""
        m = slide.shape[0]
        eps = self.lock_limit_eps
        lo = torch.full((m,), max(self._slide_range_lo, 0.0), device=self.device)
        hi = torch.full((m,), self._slide_range_hi, device=self.device)
        hi = torch.where(slide, hi, torch.full_like(hi, eps))
        return torch.stack((lo, hi), dim=-1).unsqueeze(1)

    def _mjlab_write_jnt_range(
        self,
        env_ids: torch.Tensor,
        limits_m2: torch.Tensor,
        joint_id: int | None = None,
    ) -> None:
        """Write ``jnt_range`` for one door joint (mjlab expanded model)."""
        from active_adaptation.envs.mdp.randomizations.common import (
            _mjlab_expand_model_fields,
        )

        _mjlab_expand_model_fields(self.env, "jnt_range")
        model = self.env.sim.model
        local_id = self.door_joint_id if joint_id is None else joint_id
        jid = local_id
        if hasattr(self.asset, "indexing") and hasattr(self.asset.indexing, "joint_ids"):
            jid = int(self.asset.indexing.joint_ids[local_id].item())
        model.jnt_range[env_ids.cpu(), jid, :] = limits_m2.detach().cpu()

    def _mjlab_write_door_pd_gains(
        self,
        env_ids: torch.Tensor,
        kp: torch.Tensor,
        kd: torch.Tensor,
        joint_name: str = "door_joint",
    ) -> None:
        """Best-effort: set BuiltinPd position/velocity actuator gains."""
        from active_adaptation.envs.mdp.randomizations.common import (
            _mjlab_expand_model_fields,
        )

        names = list(getattr(self.asset, "actuator_names", ()))
        try:
            pos_i = next(i for i, n in enumerate(names) if n.endswith(f"{joint_name}_pd_pos"))
            vel_i = next(i for i, n in enumerate(names) if n.endswith(f"{joint_name}_pd_vel"))
        except StopIteration:
            return
        _mjlab_expand_model_fields(self.env, "actuator_gainprm", "actuator_biasprm")
        model = self.env.sim.model
        # MuJoCo position actuator: gainprm[0]=kp, biasprm[1]=-kp.
        env_cpu = env_ids.detach().cpu()
        kp_cpu = kp.detach().cpu()
        kd_cpu = kd.detach().cpu()
        model.actuator_gainprm[env_cpu, pos_i, 0] = kp_cpu
        model.actuator_biasprm[env_cpu, pos_i, 1] = -kp_cpu
        model.actuator_gainprm[env_cpu, vel_i, 0] = kd_cpu


__all__ = [
    "DoorBehavior",
    "DIR_PULL",
    "DIR_PUSH",
    "DIR_SLIDE",
    "SLIDE_HANDLE_ANGLE",
    "OpenDirection",
]
