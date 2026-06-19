import torch

from typing import TYPE_CHECKING, Union
from typing_extensions import override

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.sensors import ContactSensor
    from active_adaptation.envs.env_base import _EnvBase

    IsaacEntity = Union[Articulation, RigidObject]

from .base import TerminationV2
from active_adaptation.envs.utils import find_sensor_bodies, find_bodies


class max_episode_length(TerminationV2):
    """Termination when episode length exceeds the specified maximum episode length."""

    def __init__(self):
        super().__init__(is_timeout=True)

    def compute(self, termination: torch.Tensor):
        cur_length = self.env.episode_length_buf.reshape(self.num_envs, 1)
        max_length = self.env.max_episode_length
        return cur_length >= max_length


class crash(TerminationV2):
    """
    Terminate when a monitored link has been in **contact** long enough, optionally with random gating.
    """

    def __init__(
        self,
        body_names_expr: str,
        t_thres: float = 0.0,
        prob: float = 1.0,
    ):
        super().__init__()
        self.body_names_expr = body_names_expr
        self.t_thres = t_thres
        self.prob = min(max(prob, 0.0), 1.0)

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: ContactSensor = self.env.scene.sensors["contact_forces"]
        self.body_indices, self.body_names = find_sensor_bodies(
            self.asset, self.contact_sensor, self.body_names_expr
        )
        self.body_indices = torch.tensor(self.body_indices, device=self.device)
        self.data = self.contact_sensor.data

    def __repr__(self) -> str:
        return (
            f"crash(expr={self.body_names_expr!r}, t_thres={self.t_thres}, "
            f"prob={self.prob}, bodies={self.body_names!r}, "
            f"indices={self.body_indices.tolist()})"
        )

    def compute(self, termination: torch.Tensor):
        contact_time = self.data.current_contact_time[:, self.body_indices]
        terminated = (contact_time > self.t_thres).any(1, True)
        if self.prob < 1.0:
            terminated = terminated & (torch.rand(self.num_envs, 1, device=self.device) <= self.prob)
            discount = torch.where(terminated, 1.0 - self.prob, 1.0)
            return terminated, discount
        return terminated


class undesired_contact(TerminationV2):
    """Soft termination based on the contact forces on the specified body names."""

    supported_backends = ("isaac", "mjlab", "motrix")

    def __init__(
        self,
        body_names: str,
        thres: float = 1.0,
        lateral_only: bool = False,
    ):
        super().__init__()
        self.body_names_pattern = body_names
        self.thres = thres
        self.lateral_only = lateral_only
        self.dim = 2 if lateral_only else 3

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: ContactSensor = self.env.scene.sensors["contact_forces"]
        self.body_indices, self.body_names = find_sensor_bodies(
            self.asset, self.contact_sensor, self.body_names_pattern
        )
        self.body_indices = torch.tensor(self.body_indices, device=self.device)

    def __repr__(self) -> str:
        return (
            f"undesired_contact(body_names={self.body_names}, "
            f"body_indices={self.body_indices.tolist()}, thres={self.thres}, "
            f"lateral_only={self.lateral_only})"
        )

    def compute(self, termination: torch.Tensor):
        terminated = torch.zeros(self.num_envs, 1, device=self.device, dtype=bool)
        contact_time = self.contact_sensor.data.current_contact_time[:, self.body_indices]
        in_contact = (contact_time > 0.0).sum(dim=1, keepdim=True)
        discount = 0.8**in_contact
        return terminated, discount.reshape(self.num_envs, 1)


class fall_over(TerminationV2):
    def __init__(
        self,
        xy_thres: float = 0.8,
        entity_name: str = "robot",
    ):
        super().__init__()
        self.xy_thres = xy_thres
        self.entity_name = entity_name

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: IsaacEntity = self.env.scene[self.entity_name]

    def compute(self, termination: torch.Tensor):
        gravity_xy: torch.Tensor = self.asset.data.projected_gravity_b[:, :2]
        fall_over = gravity_xy.norm(dim=1, keepdim=True) >= self.xy_thres
        return fall_over


class root_pos_error(TerminationV2):
    def __init__(self, threshold: float = 2.0, dim: int = 2):
        super().__init__()
        self.threshold = threshold
        self.dim = dim

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        valid = (self.env.episode_length_buf > 10).unsqueeze(-1)
        target_pos = self.command_manager.cmd_pos_w[:, : self.dim]
        pos_error = (
            (self.asset.data.root_link_pos_w[:, : self.dim] - target_pos)
            .square()
            .sum(-1, True)
        )
        return (valid & (pos_error > self.threshold)).reshape(self.num_envs, 1)


class cum_error(TerminationV2):
    def __init__(self, thres: float = 0.85, min_steps: int = 50):
        super().__init__()
        self.thres = thres
        self.min_steps = min_steps

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.thres = torch.tensor(self.thres, device=self.device)
        if not hasattr(self.command_manager, "cum_error"):
            raise ValueError("`cum_error` attribute not found in command manager")

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        return (self.command_manager.cum_error > self.thres).any(-1, True)


class joint_acc_exceeds(TerminationV2):
    def __init__(self, thres: float):
        super().__init__()
        self.thres = thres

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        valid = (self.env.episode_length_buf > 2).unsqueeze(-1)
        return valid & (self.asset.data.joint_acc.abs() > self.thres).any(1, True)


class root_height_below(TerminationV2):
    def __init__(self, thres: float):
        super().__init__()
        self.thres = thres

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        ground_height = self.env.get_ground_height_at(self.asset.data.root_pos_w)
        height = self.asset.data.root_pos_w[:, 2] - ground_height
        return (height < self.thres).reshape(self.num_envs, 1)


class force_contact(TerminationV2):
    def __init__(self, body_names: str, threshold: float):
        super().__init__()
        self.body_names_pattern = body_names
        self.threshold = threshold

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: ContactSensor = self.env.scene.sensors["contact_forces"]
        self.body_indices, self.body_names = find_sensor_bodies(
            self.asset, self.contact_sensor, self.body_names_pattern
        )

    def __repr__(self) -> str:
        return (
            f"force_contact(body_names={self.body_names}, "
            f"body_indices={self.body_indices.tolist()}, threshold={self.threshold})"
        )

    def compute(self, termination: torch.Tensor):
        forces = self.contact_sensor.data.net_forces_w[:, self.body_indices].norm(dim=-1)
        in_contact = forces.sum(dim=1, keepdim=True) > self.threshold
        return in_contact


class bodies_too_close(TerminationV2):
    """Terminate when any two of the specified bodies are closer than ``threshold`` (meters)."""

    def __init__(self, body_names: str, threshold: float = 0.05):
        super().__init__()
        self.body_names_pattern = body_names
        self.threshold = threshold

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.body_indices, self.body_names = find_bodies(self.asset, self.body_names_pattern)
        self.body_indices = torch.tensor(self.body_indices, device=self.device)
        if len(self.body_indices) < 2:
            raise ValueError("At least two bodies are required")
        n = len(self.body_indices)
        self.pair_i, self.pair_j = torch.triu_indices(n, n, offset=1)

    def __repr__(self) -> str:
        return (
            f"bodies_too_close(body_names={self.body_names}, "
            f"body_indices={self.body_indices.tolist()}, threshold={self.threshold})"
        )

    def compute(self, termination: torch.Tensor):
        body_pos_w = self.asset.data.body_pos_w[:, self.body_indices]
        dist = torch.cdist(body_pos_w, body_pos_w)
        dist = dist[:, self.pair_i, self.pair_j].reshape(self.num_envs, -1)
        return (dist < self.threshold).any(dim=-1, keepdim=True)


def _point_segment_dist_sq(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared distance from points ``p`` to segment ``a``–``b``. All tensors (B, 3)."""
    ab = b - a
    ap = p - a
    denom = (ab * ab).sum(dim=-1).clamp_min(1e-20)
    t = ((ap * ab).sum(dim=-1) / denom).clamp(0.0, 1.0)
    closest = a + t.unsqueeze(-1) * ab
    return (p - closest).square().sum(dim=-1)


@torch.compile
def _segment_segment_dist_sq(
    p1: torch.Tensor, p2: torch.Tensor, q1: torch.Tensor, q2: torch.Tensor
) -> torch.Tensor:
    """Batched shortest distance squared between segments ``p1``–``p2`` and ``q1``–``q2`` in R^3."""
    u = p2 - p1
    v = q2 - q1
    w0 = p1 - q1
    a = (u * u).sum(dim=-1)
    b = (u * v).sum(dim=-1)
    c = (v * v).sum(dim=-1)
    d = (u * w0).sum(dim=-1)
    e = (v * w0).sum(dim=-1)
    denom = a * c - b * b
    eps = 1e-20
    s = (b * e - c * d) / denom.clamp_min(eps)
    t = (a * e - b * d) / denom.clamp_min(eps)
    interior = (denom > eps) & (s >= 0.0) & (s <= 1.0) & (t >= 0.0) & (t <= 1.0)
    diff = w0 + s.unsqueeze(-1) * u - t.unsqueeze(-1) * v
    d_unc_sq = diff.square().sum(dim=-1)
    d_edge_sq = torch.minimum(
        torch.minimum(_point_segment_dist_sq(p1, q1, q2), _point_segment_dist_sq(p2, q1, q2)),
        torch.minimum(_point_segment_dist_sq(q1, p1, p2), _point_segment_dist_sq(q2, p1, p2)),
    )
    return torch.where(interior, d_unc_sq, d_edge_sq)


class segments_cross(TerminationV2):
    """Terminate when the shortest distance between the two segments is below ``threshold``."""

    def __init__(
        self,
        segment1_names: str,
        segment2_names: str,
        threshold: float = 0.07,
    ):
        super().__init__()
        self.segment1_names_pattern = segment1_names
        self.segment2_names_pattern = segment2_names
        self.threshold = threshold
        self._threshold_sq = threshold * threshold

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.segment1_indices, self.segment1_names = find_bodies(
            self.asset, self.segment1_names_pattern
        )
        self.segment1_indices = torch.tensor(self.segment1_indices, device=self.device)
        self.segment2_indices, self.segment2_names = find_bodies(
            self.asset, self.segment2_names_pattern
        )
        self.segment2_indices = torch.tensor(self.segment2_indices, device=self.device)
        if len(self.segment1_indices) != 2 or len(self.segment2_indices) != 2:
            raise ValueError("segments_cross requires exactly two bodies per segment (endpoints).")

    def __repr__(self) -> str:
        return (
            f"segments_cross(segment1={self.segment1_names}, segment2={self.segment2_names}, "
            f"threshold={self.threshold})"
        )

    def compute(self, termination: torch.Tensor):
        pos1 = self.asset.data.body_pos_w[:, self.segment1_indices]
        pos2 = self.asset.data.body_pos_w[:, self.segment2_indices]
        p1, p2 = pos1[:, 0], pos1[:, 1]
        q1, q2 = pos2[:, 0], pos2[:, 1]
        d_sq = _segment_segment_dist_sq(p1, p2, q1, q2)
        return (d_sq < self._threshold_sq).reshape(self.num_envs, 1)
