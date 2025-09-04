import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor

from active_adaptation.envs.mdp.base import Reward


class max_feet_height(Reward):
    def __init__(self, env, weight: float, body_names: str, target_height: float):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_ids = self.asset.find_bodies(body_names)[0]
        self.body_contact_ids = self.contact_sensor.find_bodies(body_names)[0]
        self.target_height = target_height

        self.max_height = torch.zeros(self.num_envs, len(self.body_ids), device=self.device)
        self.rew = torch.zeros(self.num_envs, 1, device=self.device)

    def reset(self, env_ids):
        self.max_height[env_ids] = 0.
    
    def update(self):
        feet_height = self.asset.data.body_pos_w[:, self.body_ids, 2]
        in_contact = self.contact_sensor.data.current_contact_time[:, self.body_contact_ids] > 0.0
        self.max_height = torch.maximum(self.max_height, feet_height)
        self.rew = self.max_height.clamp_max(self.target_height)
        self.max_height = torch.where(in_contact, 0., self.max_height)

    def compute(self) -> torch.Tensor:
        first_contact = self.contact_sensor.compute_first_contact(self.env.step_dt)[:, self.body_contact_ids]
        rew = self.rew * first_contact
        return rew.sum(1, keepdim=True)

