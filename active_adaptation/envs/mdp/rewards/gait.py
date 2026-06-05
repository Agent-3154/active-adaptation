import torch
from typing import TYPE_CHECKING
from typing_extensions import override

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor as IsaacContactSensor
    from mjlab.sensor import ContactSensor as MjlabContactSensor

from .base import Reward
from active_adaptation.envs.utils import find_bodies, find_sensor_bodies


class max_feet_height(Reward):
    def __init__(self, env, weight: float, body_names: str, target_height: float):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: IsaacContactSensor = self.env.scene.sensors["contact_forces"]
        self.body_ids, self.body_names = find_bodies(self.asset, body_names)
        self.body_contact_ids = find_sensor_bodies(
            self.asset, self.contact_sensor, body_names
        )[0]
        self.target_height = target_height

        self.max_height = torch.zeros(
            self.num_envs, len(self.body_ids), device=self.device
        )
        self.rew = torch.zeros(self.num_envs, 1, device=self.device)

    @override
    def reset(self, env_ids):
        self.max_height[env_ids] = 0.0

    @override
    def update(self):
        feet_height = self.asset.data.body_link_pos_w[:, self.body_ids, 2]
        in_contact = (
            self.contact_sensor.data.current_contact_time[:, self.body_contact_ids]
            > 0.0
        )
        self.max_height = torch.maximum(self.max_height, feet_height)
        self.rew = self.max_height.clamp_max(self.target_height)
        self.max_height = torch.where(in_contact, 0.0, self.max_height)

    @override
    def _compute(self) -> torch.Tensor:
        first_contact = self.contact_sensor.compute_first_contact(self.env.step_dt)[
            :, self.body_contact_ids
        ]
        rew = self.rew * first_contact
        return rew.sum(1, keepdim=True)


class feet_sliding(Reward):
    supported_backends = ("isaac", "mjlab", "motrix")

    def __init__(self, env, body_names: str, weight: float):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: IsaacContactSensor = self.env.scene.sensors["contact_forces"]
        self.contact_data = self.contact_sensor.data
        self.body_ids, self.body_names = find_bodies(self.asset, body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.body_contact_ids = find_sensor_bodies(
            self.asset, self.contact_sensor, body_names
        )[0]
        self.body_contact_ids = torch.tensor(self.body_contact_ids, device=self.device)

    @override
    def _compute(self) -> torch.Tensor:
        in_contact = (
            self.contact_data.current_contact_time[:, self.body_contact_ids]
            > self.env.physics_dt
        )
        if self.env.backend == "isaac":
            feet_speed = self.asset.data.body_com_lin_vel_w[:, self.body_ids].norm(dim=-1)
        elif self.env.backend in ("mjlab", "motrix"):
            feet_speed = self.asset.data.body_link_lin_vel_w[:, self.body_ids].norm(dim=-1)
        sliding = (in_contact * feet_speed).sum(dim=1)
        return -sliding.reshape(self.num_envs, 1)


class quadruped_trot(Reward):
    """
    Reward either (FL-RR) or (FR-RL) are in contact but not both.
    """

    def __init__(self, env, weight: float, body_names: str):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: IsaacContactSensor = self.env.scene.sensors["contact_forces"]
        self.body_ids, self.body_names = find_bodies(self.asset, body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)

        self.body_contact_ids = find_sensor_bodies(
            self.asset, self.contact_sensor, body_names
        )[0]
        self.body_contact_ids = torch.tensor(self.body_contact_ids, device=self.device)

    @override
    def _compute(self) -> torch.Tensor:
        in_contact = (
            self.contact_sensor.data.current_contact_time[:, self.body_contact_ids]
            > 0.005
        )
        FL_RR = in_contact[:, [0, 3]].all(dim=1)
        FR_RL = in_contact[:, [1, 2]].all(dim=1)
        rew = torch.logical_xor(FL_RR, FR_RL)
        active = ~self.command_manager.is_standing_env
        return rew.reshape(self.num_envs, 1), active.reshape(self.num_envs, 1)


class feet_clearance(Reward):
    """
    Smooth penalty for feet getting too close.

    Pairwise distances between foot bodies are computed per environment (upper-triangular
    pairs to avoid double counting). Distances larger than `thres` saturate to zero
    penalty; distances below `thres` yield negative reward via a log distance ratio.
    """

    def __init__(self, env, body_names: str, weight: float, thres: float = 0.1):
        super().__init__(env, weight)
        self.thres = thres
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.body_ids, self.body_names = find_bodies(self.asset, body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.num_feet = len(self.body_ids)

    @override
    def _compute(self) -> torch.Tensor:
        feet_pos_w = self.asset.data.body_link_pos_w[:, self.body_ids]
        pairwise_distances = (
            feet_pos_w.reshape(self.num_envs, 1, self.num_feet, 3)
            - feet_pos_w.reshape(self.num_envs, self.num_feet, 1, 3)
        ).norm(dim=-1)
        # Continuous penalty: use a sigmoid to smoothly approximate the hard threshold.
        # For each pair (i, j), closeness ~ 1 when d << thres, ~ 0 when d >> thres.
        distances = pairwise_distances.triu(diagonal=1).reshape(self.num_envs, -1)
        reward = (distances / self.thres).clamp_max(1.0).log().sum(dim=1, keepdim=True)
        return reward


class feet_air_time(Reward):
    def __init__(
        self,
        env,
        body_names: str,
        thres: float,
        weight: float,
        track_var: bool = False,
    ):
        super().__init__(env, weight, track_var=track_var)
        self.thres = thres
        self.asset: Articulation = self.env.scene.articulations["robot"]

        self.articulation_body_ids, self.body_names = find_bodies(self.asset, body_names)
        self.contact_sensor: IsaacContactSensor = self.env.scene.sensors["contact_forces"]
        self.body_ids = find_sensor_bodies(
            self.asset, self.contact_sensor, body_names
        )[0]
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)

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
    supported_backends = ("isaac", "mjlab", "motrix")

    def __init__(self, env, body_names: str, weight: float):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: IsaacContactSensor = self.env.scene.sensors["contact_forces"]

        self.articulation_body_ids, self.body_names = find_bodies(self.asset, body_names)
        self.body_ids = find_sensor_bodies(
            self.asset, self.contact_sensor, body_names
        )[0]
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)
        self.first_contact = torch.zeros(
            self.num_envs, len(self.body_ids), device=self.env.device
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
        env,
        body_names: str,
        margin: float,
        weight: float,
        track_var: bool = False,
    ):
        super().__init__(env, weight, track_var=track_var)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: IsaacContactSensor = self.env.scene.sensors["contact_forces"]
        self.body_ids, self.body_names = find_sensor_bodies(self.asset, self.contact_sensor, body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.margin = margin

    @override
    def _compute(self) -> torch.Tensor:
        in_contact = self.contact_sensor.data.current_contact_time[:, self.body_ids] > self.margin
        single_contact = torch.where(torch.sum(in_contact, dim=1) == 1, 0., -1.)
        valid = ~self.command_manager.is_standing_env
        return single_contact.reshape(self.num_envs, 1), valid.reshape(self.num_envs, 1)

