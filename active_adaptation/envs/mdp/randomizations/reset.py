"""Episode-reset randomizations for articulation joint state.

These terms run in :meth:`Randomization.reset` *after* ``Command.reset``, so they
overwrite whatever joint pose the command wrote at spawn. Use them to diversify
the start of each episode; they do not resample mid-episode.

Choose a term by how the sample interval is defined around the default pose
``q0`` (with hard limits ``[q_min, q_max]``):

``reset_joint_states_uniform``
    Sample in a **configured** interval. With ``rel=False`` (default) the YAML
    ranges are absolute joint positions. With ``rel=True`` they are offsets
    added to ``q0`` (typical: ``pos_ranges: {.*: [-0.1, 0.1]}``).

``reset_joint_states_scale``
    Sample by **scaling** ``q0``: ``q ~ U(low, high) * q0``. Useful when defaults
    are away from zero (quadruped legs). Note that joints with ``q0 ≈ 0`` barely
    move; use uniform or limit-frac for those.

``reset_joint_states_limit_frac``
    Sample in an interval that **interpolates from ``q0`` toward the limits**:

        ``[q0 - frac (q0 - q_min),  q0 + frac (q_max - q0)]``

    ``frac=0`` is always the default pose; ``frac=1`` is uniform over the full
    limit range. Unlike scale, this still randomizes joints whose default is zero.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import torch
from tensordict import TensorDictBase
from typing_extensions import override

import active_adaptation.utils.string as string_utils
from active_adaptation.envs.utils import find_joints

from .base import Randomization
from .common import sample_uniform

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


def _resolve_joint_values(asset, spec: Dict[str, tuple]):
    """Match a name→value dict against simulation joint order, then asset indices."""
    _, joint_names, values = string_utils.resolve_matching_names_values(
        dict(spec), asset.cfg.joint_names_simulation
    )
    joint_ids = [asset.joint_names.index(name) for name in joint_names]
    return joint_ids, joint_names, values


class reset_joint_states_uniform(Randomization):
    """Reset joints by sampling positions uniformly in configured ranges.

    Args:
        pos_ranges: Mapping from joint-name regex to ``(low, high)``.
        vel_ranges: Optional mapping for joint velocities. When omitted, velocity
            is the default joint velocity (usually zero).
        rel: If ``True``, sampled positions are added to the default pose.
            If ``False``, ranges are absolute joint coordinates.
        entity_name: Scene articulation to reset (default ``"robot"``).
    """

    def __init__(
        self,
        pos_ranges: Dict[str, tuple],
        vel_ranges: Dict[str, tuple] | None = None,
        rel: bool = False,
        entity_name: str = "robot",
    ):
        super().__init__()
        self.pos_ranges = pos_ranges
        self.vel_ranges = vel_ranges
        self.rel = rel
        self.entity_name = entity_name

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset = self.env.scene.articulations[self.entity_name]
        self.joint_ids, _, pos_ranges = _resolve_joint_values(self.asset, self.pos_ranges)
        self.pos_ranges = torch.as_tensor(pos_ranges, device=self.device).unbind(-1)
        if self.vel_ranges is not None:
            _, _, vel_ranges = _resolve_joint_values(self.asset, self.vel_ranges)
            self.vel_ranges = torch.as_tensor(vel_ranges, device=self.device).unbind(-1)
        else:
            self.vel_ranges = None
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids].float()
        self.default_joint_vel = self.asset.data.default_joint_vel[:, self.joint_ids].float()
        self.joint_limits = self.asset.data.joint_pos_limits[0, self.joint_ids].float().unbind(-1)

    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase):
        if env_ids.numel() == 0:
            return
        shape = (len(env_ids), len(self.joint_ids))
        init_pos = sample_uniform(shape, *self.pos_ranges, self.device)
        if self.rel:
            init_pos += self.default_joint_pos[env_ids]
        if self.vel_ranges is not None:
            init_vel = sample_uniform(shape, *self.vel_ranges, self.device)
        else:
            init_vel = torch.zeros(shape, device=self.device)
        init_vel += self.default_joint_vel[env_ids]
        self.asset.write_joint_state_to_sim(
            init_pos.clamp(*self.joint_limits),
            init_vel,
            self.joint_ids,
            env_ids,
        )


class reset_joint_states_scale(Randomization):
    """Reset joints by independently scaling each default position.

    Sampled pose is ``q = u * q0`` with ``u ~ Uniform(low, high)`` per joint
    (from ``pos_scales``). Velocities stay at their defaults.

    Args:
        pos_scales: Mapping from joint-name regex to ``(low, high)`` scale factors.
        entity_name: Scene articulation to reset (default ``"robot"``).
    """

    def __init__(self, pos_scales: Dict[str, tuple], entity_name: str = "robot"):
        super().__init__()
        self.pos_scales = pos_scales
        self.entity_name = entity_name

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset = self.env.scene.articulations[self.entity_name]
        self.joint_ids, _, pos_scales = _resolve_joint_values(self.asset, self.pos_scales)
        self.pos_scales = torch.as_tensor(pos_scales, device=self.device).unbind(-1)
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids]
        self.default_joint_vel = self.asset.data.default_joint_vel[:, self.joint_ids]

    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase):
        if env_ids.numel() == 0:
            return
        low, high = self.pos_scales
        default_pos = self.default_joint_pos[env_ids]
        scale = torch.rand_like(default_pos) * (high - low) + low
        init_pos = default_pos * scale
        init_vel = self.default_joint_vel[env_ids]
        self.asset.write_joint_state_to_sim(init_pos, init_vel, self.joint_ids, env_ids)


class reset_joint_states_limit_frac(Randomization):
    """Reset joints by sampling a fraction of the way from default toward limits.

    For each joint the sample box is

        ``[q0 - frac (q0 - q_min),  q0 + frac (q_max - q0)]``

    so ``frac`` is a dimensionless blend: ``0`` keeps the default pose, ``1``
    uses the full ``joint_pos_limits`` interval. Positions are clamped to those
    limits. Velocities stay at their defaults.

    Args:
        frac: Fraction of the default-to-limit gap used as half-width of the
            sample interval. Must be in ``[0, 1]``.
        joint_names: Regex or list of regexes for joints to randomize.
        entity_name: Scene articulation to reset (default ``"robot"``).
    """

    def __init__(
        self,
        frac: float = 0.5,
        joint_names: str | list[str] = ".*",
        entity_name: str = "robot",
    ):
        super().__init__()
        if not 0.0 <= frac <= 1.0:
            raise ValueError(f"frac must be in [0, 1], got {frac}")
        self.frac = float(frac)
        self.joint_names_pattern = joint_names
        self.entity_name = entity_name

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset = self.env.scene.articulations[self.entity_name]
        self.joint_ids, self.joint_names = find_joints(self.asset, self.joint_names_pattern)
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids].float()
        self.default_joint_vel = self.asset.data.default_joint_vel[:, self.joint_ids].float()
        limits = self.asset.data.joint_pos_limits[0, self.joint_ids].float()
        self.limit_low, self.limit_high = limits.unbind(-1)

    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase):
        if env_ids.numel() == 0:
            return
        q0 = self.default_joint_pos[env_ids]
        low = q0 - self.frac * (q0 - self.limit_low)
        high = q0 + self.frac * (self.limit_high - q0)
        init_pos = sample_uniform(q0.shape, low, high, self.device).clamp(
            self.limit_low, self.limit_high
        )
        init_vel = self.default_joint_vel[env_ids]
        self.asset.write_joint_state_to_sim(init_pos, init_vel, self.joint_ids, env_ids)
