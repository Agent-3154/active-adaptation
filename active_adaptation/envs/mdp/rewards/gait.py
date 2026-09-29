import torch
from typing import TYPE_CHECKING
from typing_extensions import override
from tensordict import TensorDictBase

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor as IsaacContactSensor
    from mjlab.sensor import ContactSensor as MjlabContactSensor
    from active_adaptation.envs.env_base import EnvBase

from .base import Reward
from active_adaptation.envs.utils import find_bodies, find_sensor_bodies


class max_swing_height(Reward):
    def __init__(self, weight: float, body_names: str, target_height: float):
        super().__init__(weight)
        self.body_names_pattern = body_names
        self.target_height = target_height

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: IsaacContactSensor = self.env.scene.sensors["contact_forces"]
        self.body_ids, self.body_names = find_bodies(self.asset, self.body_names_pattern)
        self.body_contact_ids = find_sensor_bodies(
            self.asset, self.contact_sensor, self.body_names_pattern
        )[0]
        self.max_height = torch.zeros(
            self.num_envs, len(self.body_ids), device=self.device
        )
        self.rew = torch.zeros(self.num_envs, 1, device=self.device)

    @override
    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase):
        self.max_height[env_ids] = 0.0

    @override
    def _update(self):
        feet_height = self.asset.data.body_link_pos_w[:, self.body_ids, 2]
        self.max_height = torch.maximum(self.max_height, feet_height).clamp_max(self.target_height)
        self.first_contact = self.contact_sensor.compute_first_contact(self.env.step_dt)[
            :, self.body_contact_ids
        ]
        self.rew = (self.first_contact * self.max_height).sum(1, keepdim=True)
        self.max_height = torch.where(self.first_contact, 0.0, self.max_height)

    @override
    def _compute(self) -> torch.Tensor:
        active = (~self.command_manager.is_standing_env) & self.first_contact.any(dim=1, keepdim=True)
        return self.rew.reshape(self.num_envs, 1), active.reshape(self.num_envs, 1)


class feet_sliding(Reward):
    supported_backends = ("isaaclab", "mjlab", "motrix")

    def __init__(self, body_names: str, weight: float):
        super().__init__(weight)
        self.body_names_pattern = body_names

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: IsaacContactSensor = self.env.scene.sensors["contact_forces"]
        self.contact_data = self.contact_sensor.data
        self.body_ids, self.body_names = find_bodies(self.asset, self.body_names_pattern)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.body_contact_ids = find_sensor_bodies(
            self.asset, self.contact_sensor, self.body_names_pattern
        )[0]
        self.body_contact_ids = torch.tensor(self.body_contact_ids, device=self.device)

    @override
    def _compute(self) -> torch.Tensor:
        in_contact = (
            self.contact_data.current_contact_time[:, self.body_contact_ids]
            > self.env.physics_dt
        )
        if self.env.backend == "isaaclab":
            feet_speed = self.asset.data.body_com_lin_vel_w[:, self.body_ids].norm(dim=-1)
        elif self.env.backend in ("mjlab", "motrix"):
            feet_speed = self.asset.data.body_link_lin_vel_w[:, self.body_ids].norm(dim=-1)
        sliding = (in_contact * feet_speed).sum(dim=1)
        return -sliding.reshape(self.num_envs, 1)


# Body-name prefix → trot role. Hind feet are HL/HR in the paper and RL/RR on our assets.
_FOOT_ROLE = {
    "FL": "FL",
    "LF": "FL",
    "FR": "FR",
    "RF": "FR",
    "HL": "HL",
    "RL": "HL",
    "LH": "HL",
    "HR": "HR",
    "RR": "HR",
    "RH": "HR",
}
_TROT_ROLES = ("FL", "FR", "HL", "HR")


def _quadruped_foot_order(body_names: list[str]) -> list[int]:
    """Column order FL, FR, HL, HR inside a foot-pattern match."""
    columns: dict[str, int] = {}
    for index, name in enumerate(body_names):
        role = _FOOT_ROLE.get(name.split("_", 1)[0].upper())
        if role is None:
            raise ValueError(
                f"Cannot map foot {name!r} to FL/FR/HL/HR. "
                f"Expected a name starting with one of {sorted(_FOOT_ROLE)}."
            )
        if role in columns:
            raise ValueError(f"Duplicate {role} foot in {body_names}")
        columns[role] = index
    missing = [role for role in _TROT_ROLES if role not in columns]
    if missing:
        raise ValueError(f"Need feet {list(_TROT_ROLES)}, missing {missing} in {body_names}")
    return [columns[role] for role in _TROT_ROLES]


def four_leg_trot_cost(t_c: torch.Tensor, t_a: torch.Tensor) -> torch.Tensor:
    """Mismatch cost (D.1). ``t_c`` / ``t_a`` are ``(N, 4)`` in order FL, FR, HL, HR.

    Penalizes contact and air duration differences between diagonal pairs
    (FL–HR, FR–HL), and between each foot and the air duration of its
    non-diagonal counterpart.
    """
    fl, fr, hl, hr = 0, 1, 2, 3
    cost = (
        (t_c[:, fl] - t_c[:, hr]).abs()
        + (t_c[:, fr] - t_c[:, hl]).abs()
        + (t_a[:, fl] - t_a[:, hr]).abs()
        + (t_a[:, fr] - t_a[:, hl]).abs()
        + (t_c[:, fl] - t_a[:, fr]).abs()
        + (t_c[:, hl] - t_a[:, hr]).abs()
        + (t_a[:, fl] - t_c[:, fr]).abs()
        + (t_a[:, hl] - t_c[:, hr]).abs()
    )
    return cost.unsqueeze(-1)


def _phase_durations(
    current_contact: torch.Tensor,
    last_contact: torch.Tensor,
    current_air: torch.Tensor,
    last_air: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contact and air durations: the clock of the active phase, else the last completed one."""
    in_contact = current_contact > 0
    t_c = torch.where(in_contact, current_contact, last_contact)
    t_a = torch.where(in_contact, last_air, current_air)
    return t_c, t_a


def _set_dynamic_weight(reward: Reward, weight: torch.Tensor | None) -> None:
    """Cache a per-env scale from ``in_keys``. A missing key leaves the scale at 1."""
    if weight is None:
        reward._weight = torch.ones(reward.num_envs, 1, device=reward.device)
    else:
        reward._weight = weight.reshape(reward.num_envs, 1)


def _current_phase(
    current_contact: torch.Tensor,
    current_air: torch.Tensor,
    contact_eps: float,
) -> torch.Tensor:
    """Time each foot has spent in its current contact or swing phase."""
    in_contact = current_contact > contact_eps
    return torch.where(in_contact, current_contact, current_air)


def _phase_overtime(phase: torch.Tensor, max_phase: float) -> torch.Tensor:
    """Seconds each foot has stayed in the current phase past ``max_phase``, summed."""
    return (phase - max_phase).clamp(min=0).sum(dim=-1, keepdim=True)


def _trot_active(reward: Reward) -> torch.Tensor:
    standing = reward.command_manager.is_standing_env.reshape(reward.num_envs, 1)
    return (~standing) & (reward._weight > 0.0)


def _init_quadruped_feet(reward: Reward, body_names_pattern: str) -> None:
    reward.asset = reward.env.scene.articulations["robot"]
    reward.contact_sensor = reward.env.scene.sensors["contact_forces"]
    contact_ids, body_names = find_sensor_bodies(
        reward.asset, reward.contact_sensor, body_names_pattern
    )
    reward.body_names = body_names
    reward.body_contact_ids = torch.as_tensor(
        contact_ids, device=reward.device, dtype=torch.long
    )
    reward.foot_order = torch.as_tensor(
        _quadruped_foot_order(body_names),
        device=reward.device,
        dtype=torch.long,
    )


class quadruped_trot_binary(Reward):
    """Reward exactly one diagonal pair in contact: FL–HR or FR–HL, not both.

    Feet are identified by name (``RL``/``RR`` count as hind left/right), not by
    match order. The bonus stops once any foot has held its current phase longer
    than ``max_phase``, so a frozen diagonal stance does not keep paying.
    ``quadruped_trot_binary_weight`` on the step tensordict scales the reward
    per environment; a missing key leaves the scale at 1. Inactive while
    standing or while that weight is not positive.
    """

    in_keys = ["quadruped_trot_binary_weight"]
    out_keys = None

    def __init__(self, weight: float, body_names: str, max_phase: float = 0.8):
        super().__init__(weight)
        self.body_names_pattern = body_names
        self.max_phase = float(max_phase)
        if self.max_phase <= 0.0:
            raise ValueError(f"max_phase must be > 0, got {max_phase}")

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        _init_quadruped_feet(self, self.body_names_pattern)
        self._weight = torch.ones(self.num_envs, 1, device=self.device)

    @override
    def _update(self, weight: torch.Tensor | None) -> None:
        _set_dynamic_weight(self, weight)

    @override
    def _compute(self) -> torch.Tensor:
        current_contact = self.contact_sensor.data.current_contact_time[:, self.body_contact_ids]
        current_air = self.contact_sensor.data.current_air_time[:, self.body_contact_ids]
        in_contact = current_contact > self.env.physics_dt
        ordered = in_contact[:, self.foot_order]
        fl_hr = ordered[:, 0] & ordered[:, 3]
        fr_hl = ordered[:, 1] & ordered[:, 2]
        phase = _current_phase(current_contact, current_air, self.env.physics_dt)
        stepping = (phase <= self.max_phase).all(dim=-1, keepdim=True)
        rew = torch.logical_xor(fl_hr, fr_hl).reshape(self.num_envs, 1).float()
        return rew * stepping.float() * self._weight, _trot_active(self)


class quadruped_trot_timing(Reward):
    """Negative four-leg trot cost from contact and air durations.

    ``t_c`` / ``t_a`` are the ongoing phase clock when that phase is active, and
    the last completed duration otherwise. The cost is (D.1); this term returns
    its negation so a positive ``weight`` encourages a symmetric trot.
    Time spent in the current phase past ``max_phase`` is added to the cost,
    so a frozen diagonal stance is penalized even when its clocks still match.
    ``quadruped_trot_timing_weight`` on the step tensordict scales the reward
    per environment; a missing key leaves the scale at 1. Inactive while
    standing or while that weight is not positive.
    """

    in_keys = ["quadruped_trot_timing_weight"]
    out_keys = None

    def __init__(self, weight: float, body_names: str, max_phase: float = 0.8):
        super().__init__(weight)
        self.body_names_pattern = body_names
        self.max_phase = float(max_phase)
        if self.max_phase <= 0.0:
            raise ValueError(f"max_phase must be > 0, got {max_phase}")

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        _init_quadruped_feet(self, self.body_names_pattern)
        self._weight = torch.ones(self.num_envs, 1, device=self.device)

    @override
    def _update(self, weight: torch.Tensor | None) -> None:
        _set_dynamic_weight(self, weight)

    @override
    def _compute(self) -> torch.Tensor:
        data = self.contact_sensor.data
        ids = self.body_contact_ids
        t_c, t_a = _phase_durations(
            data.current_contact_time[:, ids],
            data.last_contact_time[:, ids],
            data.current_air_time[:, ids],
            data.last_air_time[:, ids],
        )
        cost = four_leg_trot_cost(t_c[:, self.foot_order], t_a[:, self.foot_order])
        phase = _current_phase(
            data.current_contact_time[:, ids],
            data.current_air_time[:, ids],
            contact_eps=0.0,
        )
        cost = cost + _phase_overtime(phase, self.max_phase)
        return -cost * self._weight, _trot_active(self)


class feet_clearance(Reward):
    """
    Smooth penalty for feet getting too close.

    Pairwise distances between foot bodies are computed per environment (upper-triangular
    pairs to avoid double counting). Distances larger than `thres` saturate to zero
    penalty; distances below `thres` yield negative reward via a log distance ratio.
    """

    def __init__(self, body_names: str, weight: float, thres: float = 0.1):
        super().__init__(weight)
        self.body_names_pattern = body_names
        self.thres = thres

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.body_ids, self.body_names = find_bodies(self.asset, self.body_names_pattern)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.num_feet = len(self.body_ids)

    @override
    def _compute(self) -> torch.Tensor:
        feet_pos_w = self.asset.data.body_link_pos_w[:, self.body_ids]
        pairwise_distances = (
            feet_pos_w.reshape(self.num_envs, 1, self.num_feet, 3)
            - feet_pos_w.reshape(self.num_envs, self.num_feet, 1, 3)
        ).norm(dim=-1)
        distances = pairwise_distances.triu(diagonal=1).reshape(self.num_envs, -1)
        reward = (distances / self.thres).clamp_max(1.0).log().sum(dim=1, keepdim=True)
        return reward


class feet_air_time(Reward):
    def __init__(
        self,
        body_names: str,
        thres: float,
        weight: float,
        track_var: bool = False,
    ):
        super().__init__(weight, track_var=track_var)
        self.body_names_pattern = body_names
        self.thres = thres

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]

        self.articulation_body_ids, self.body_names = find_bodies(
            self.asset, self.body_names_pattern
        )
        self.contact_sensor: IsaacContactSensor = self.env.scene.sensors["contact_forces"]
        self.body_ids = find_sensor_bodies(
            self.asset, self.contact_sensor, self.body_names_pattern
        )[0]
        self.body_ids = torch.tensor(self.body_ids, device=self.device)

    @override
    def _compute(self):
        first_contact = self.contact_sensor.compute_first_contact(self.env.step_dt)[
            :, self.body_ids
        ]
        last_air_time = self.contact_sensor.data.last_air_time[:, self.body_ids]
        reward = ((last_air_time - self.thres).clamp_max(0.0) * first_contact).sum(1)
        active = ~self.command_manager.is_standing_env
        return reward.reshape(self.num_envs, 1), active


class feet_contact_count(Reward):
    supported_backends = ("isaaclab", "mjlab", "motrix")

    def __init__(self, body_names: str, weight: float):
        super().__init__(weight)
        self.body_names_pattern = body_names

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: IsaacContactSensor = self.env.scene.sensors["contact_forces"]

        self.articulation_body_ids, self.body_names = find_bodies(
            self.asset, self.body_names_pattern
        )
        self.body_ids = find_sensor_bodies(
            self.asset, self.contact_sensor, self.body_names_pattern
        )[0]
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.first_contact = torch.zeros(
            self.num_envs, len(self.body_ids), device=self.device
        )

    @override
    def _compute(self):
        self.first_contact = self.contact_sensor.compute_first_contact(
            self.env.step_dt
        )[:, self.body_ids]
        return self.first_contact.sum(1, keepdim=True)


class single_foot_contact(Reward):
    """Reward for single foot contact. Useful for bi-pedal locomotion."""

    def __init__(
        self,
        body_names: str,
        margin: float,
        weight: float,
        track_var: bool = False,
    ):
        super().__init__(weight, track_var=track_var)
        self.body_names_pattern = body_names
        self.margin = margin

    @override
    def _initialize(self, env: "EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: IsaacContactSensor = self.env.scene.sensors["contact_forces"]
        self.body_ids, self.body_names = find_sensor_bodies(
            self.asset, self.contact_sensor, self.body_names_pattern
        )
        self.body_ids = torch.tensor(self.body_ids, device=self.device)

    @override
    def _compute(self) -> torch.Tensor:
        in_contact = self.contact_sensor.data.current_contact_time[:, self.body_ids] > self.margin
        single_contact = torch.where(torch.sum(in_contact, dim=1) == 1, 0.0, -1.0)
        valid = ~self.command_manager.is_standing_env
        return single_contact.reshape(self.num_envs, 1), valid.reshape(self.num_envs, 1)
