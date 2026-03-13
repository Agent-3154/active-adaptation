import torch
from typing import TYPE_CHECKING
from .base import Observation
from active_adaptation.utils.math import quat_rotate_inverse
from active_adaptation.utils.symmetry import cartesian_space_symmetry


if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor


class last_contact_pos(Observation):
    def __init__(self, env, body_names: str, world_frame: bool=False):
        super().__init__(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.world_frame = world_frame
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_ids = self.asset.find_bodies(body_names)[0]

        self.contact_ids = self.contact_sensor.find_bodies(body_names)[0]

        with torch.device(self.device):
            self.body_ids = torch.as_tensor(self.body_ids)
            self.contact_ids = torch.as_tensor(self.contact_ids)
            self.has_contact = torch.zeros(self.num_envs, len(self.body_ids), dtype=bool)
            self.last_contact_pos_w = torch.zeros(self.num_envs, len(self.contact_ids), 3)
        self.update()
        
    def reset(self, env_ids: torch.Tensor):
        self.has_contact[env_ids] = False
    
    def update(self):
        # in_contact = self.contact_sensor.data.net_forces_w[:, self.contact_ids].norm(dim=-1) > 0.1
        in_contact = self.contact_sensor.data.current_contact_time[:, self.contact_ids] > 0.0
        self.body_link_pos_w = self.asset.data.body_link_pos_w[:, self.body_ids]

        self.has_contact.logical_or_(in_contact)
        self.last_contact_pos_w = torch.where(
            in_contact.unsqueeze(-1),
            self.body_link_pos_w,
            self.last_contact_pos_w
        )
    
    def compute(self):
        self.root_link_quat_w = self.asset.data.root_link_quat_w
        self.root_link_pos_w = self.asset.data.root_pos_w
        if self.world_frame:
            result =  self.last_contact_pos_w
        else:
            result = quat_rotate_inverse(
                self.root_link_quat_w.reshape(self.num_envs, 1, 4),
                self.last_contact_pos_w - self.root_link_pos_w.reshape(self.num_envs, 1, 3)
            )
        return result.reshape(self.num_envs, -1)

    def debug_draw(self):
        if self.env.sim.has_gui() and self.env.backend == "isaac":
            self.env.debug_draw.vector(
                self.body_link_pos_w,
                self.last_contact_pos_w - self.body_link_pos_w,
                color=(0, 0, 1, 1)
            )


class contact_indicator(Observation):
    supported_backends = ("isaac",)
    def __init__(self, env, body_names: str):
        super().__init__(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_names = self.asset.find_bodies(body_names)[1]
        self.body_ids = self.contact_sensor.find_bodies(body_names)[0]
        
    def compute(self):
        return self.contact_sensor.data.current_contact_time[:, self.body_ids] > 0.0

    def symmetry_transform(self):
        return cartesian_space_symmetry(self.asset, self.body_names, sign=(1,))


class contact_forces(Observation):
    def __init__(self, env, body_names: str, world_frame: bool=False):
        super().__init__(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.world_frame = world_frame
        self.body_names = self.asset.find_bodies(body_names)[1]
        self.body_ids = self.contact_sensor.find_bodies(body_names)[0]

    def compute(self):
        self.root_link_quat_w = self.asset.data.root_link_quat_w
        contact_forces = self.contact_sensor.data.net_forces_w[:, self.body_ids]
        if not self.world_frame:
            contact_forces = quat_rotate_inverse(
                self.root_link_quat_w.reshape(self.num_envs, 1, 4),
                contact_forces
            )
        return contact_forces.reshape(self.num_envs, -1)

    def symmetry_transform(self):
        return cartesian_space_symmetry(self.asset, self.body_names)
