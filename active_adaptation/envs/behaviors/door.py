"""Door articulation behavior: lock, open direction, handle unlock.

Attach via ``AssetSpec(behaviors=(DoorBehavior(), ...))``. Lookup with
``env.require_behavior("door.door")`` when the YAML object key is ``door``.

**Locking:** while locked, the hinge (``door_joint``) is held at ``0`` with high
stiffness (Isaac) and/or near-zero joint limits. Unlock when
``|handle_joint|`` exceeds the threshold (default 30°).

**Open direction** (object frame: hinge +Z, panel +X, through-door +Y):

- ``"pull"`` (+1): allow ``door_joint >= 0`` (panel swings toward **+Y**)
- ``"push"`` (−1): allow ``door_joint <= 0`` (panel swings toward **−Y**)
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

OpenDirection = Literal["push", "pull"]

# Signed open direction in the door frame (+ = pull / +Y, − = push / −Y).
DIR_PULL = 1
DIR_PUSH = -1


def _parse_open_direction(direction: OpenDirection | int | str) -> int:
    if isinstance(direction, int):
        if direction not in (DIR_PULL, DIR_PUSH):
            raise ValueError(f"open_direction int must be ±1, got {direction}")
        return direction
    key = str(direction).lower()
    if key == "pull":
        return DIR_PULL
    if key == "push":
        return DIR_PUSH
    raise ValueError(f"open_direction must be 'push' or 'pull', got {direction!r}")


class DoorBehavior(EntityBehavior):
    """Control door lock, open direction, and handle-based unlock."""

    name = "door"

    def __init__(
        self,
        door_joint_name: str = "door_joint",
        handle_joint_name: str = "handle_joint",
        *,
        locked_stiffness: float = 500.0,
        unlocked_stiffness: float = 0.0,
        locked_damping: float = 50.0,
        unlocked_damping: float = 4.0,
        handle_unlock_threshold_deg: float = 30.0,
        open_direction: OpenDirection = "pull",
        initially_locked: bool = True,
        lock_limit_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.door_joint_name_cfg = door_joint_name
        self.handle_joint_name_cfg = handle_joint_name
        self.locked_stiffness = float(locked_stiffness)
        self.unlocked_stiffness = float(unlocked_stiffness)
        self.locked_damping = float(locked_damping)
        self.unlocked_damping = float(unlocked_damping)
        self.handle_unlock_threshold = math.radians(float(handle_unlock_threshold_deg))
        self.default_open_direction = _parse_open_direction(open_direction)
        self.initially_locked = bool(initially_locked)
        self.lock_limit_eps = float(lock_limit_eps)

        self.door_joint_id: int = -1
        self.handle_joint_id: int = -1
        self._door_range_lo: float = -math.pi
        self._door_range_hi: float = math.pi

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

        limits = self.asset.data.joint_pos_limits[0, self.door_joint_id]
        self._door_range_lo = float(limits[0].item())
        self._door_range_hi = float(limits[1].item())

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
        """Set push/pull per env (see module docstring)."""
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
        """Auto-unlock when the handle rotates past the threshold."""
        assert self.locked is not None
        unlocked_by_handle = (
            self.handle_joint_pos.abs() >= self.handle_unlock_threshold
        )
        newly = self.locked & unlocked_by_handle
        if newly.any():
            self.locked = self.locked & ~newly
            self._gains_dirty = True
            self._apply_lock_state(newly.nonzero().squeeze(-1))

    @override
    def pre_step(self, substep: int) -> None:
        """Hold locked hinges at 0; refresh gains if dirty."""
        del substep
        assert self.locked is not None
        if self._gains_dirty:
            self._apply_lock_state()
        if not self.locked.any():
            return
        locked_ids = self.locked.nonzero().squeeze(-1)
        if locked_ids.numel() == 0:
            return
        zeros = torch.zeros(locked_ids.numel(), 1, device=self.device)
        joint_ids = [self.door_joint_id]
        self.asset.set_joint_position_target(
            zeros, joint_ids=joint_ids, env_ids=locked_ids
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
        limits = self._door_limits_for_state(locked, open_dir)
        joint_ids = [self.door_joint_id]

        if hasattr(self.asset, "write_joint_position_limit_to_sim"):
            self.asset.write_joint_position_limit_to_sim(
                limits, joint_ids=joint_ids, env_ids=env_ids, warn_limit_violation=False
            )
        elif self.env.backend == "mjlab":
            self._mjlab_write_jnt_range(env_ids, limits.squeeze(1))

        # Stiffness / damping (Isaac ImplicitActuator; best-effort elsewhere).
        stiff = torch.where(
            locked,
            torch.full((env_ids.numel(),), self.locked_stiffness, device=self.device),
            torch.full((env_ids.numel(),), self.unlocked_stiffness, device=self.device),
        ).unsqueeze(-1)
        damp = torch.where(
            locked,
            torch.full((env_ids.numel(),), self.locked_damping, device=self.device),
            torch.full((env_ids.numel(),), self.unlocked_damping, device=self.device),
        ).unsqueeze(-1)

        if hasattr(self.asset, "write_joint_stiffness_to_sim"):
            self.asset.write_joint_stiffness_to_sim(
                stiff, joint_ids=joint_ids, env_ids=env_ids
            )
            self.asset.write_joint_damping_to_sim(
                damp, joint_ids=joint_ids, env_ids=env_ids
            )
        elif self.env.backend == "mjlab":
            self._mjlab_write_door_pd_gains(env_ids, stiff.squeeze(-1), damp.squeeze(-1))

        self._gains_dirty = False

    def _mjlab_write_jnt_range(
        self, env_ids: torch.Tensor, limits_m2: torch.Tensor
    ) -> None:
        """Write ``jnt_range`` for the door hinge (mjlab expanded model)."""
        from active_adaptation.envs.mdp.randomizations.common import (
            _mjlab_expand_model_fields,
        )

        _mjlab_expand_model_fields(self.env, "jnt_range")
        model = self.env.sim.model
        # Entity-local joint id → global mj joint id via indexing if present.
        jid = self.door_joint_id
        if hasattr(self.asset, "indexing") and hasattr(self.asset.indexing, "joint_ids"):
            jid = int(self.asset.indexing.joint_ids[self.door_joint_id].item())
        model.jnt_range[env_ids.cpu(), jid, :] = limits_m2.detach().cpu()

    def _mjlab_write_door_pd_gains(
        self,
        env_ids: torch.Tensor,
        kp: torch.Tensor,
        kd: torch.Tensor,
    ) -> None:
        """Best-effort: set BuiltinPd position/velocity actuator gains."""
        from active_adaptation.envs.mdp.randomizations.common import (
            _mjlab_expand_model_fields,
        )

        names = list(getattr(self.asset, "actuator_names", ()))
        try:
            pos_i = next(i for i, n in enumerate(names) if n.endswith("door_joint_pd_pos"))
            vel_i = next(i for i, n in enumerate(names) if n.endswith("door_joint_pd_vel"))
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


__all__ = ["DoorBehavior", "DIR_PULL", "DIR_PUSH", "OpenDirection"]
