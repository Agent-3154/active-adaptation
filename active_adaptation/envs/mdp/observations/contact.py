import torch
from typing import TYPE_CHECKING
from .base import Observation
from active_adaptation.utils.math import quat_rotate_inverse
from active_adaptation.utils.symmetry import cartesian_space_symmetry


if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor


class last_contact_pos(Observation):
    """Buffered body link positions tied to contact, updated every step while the foot is down.

    Unlike touchdown-only snapshots, ``last_contact_pos_w`` follows
    ``body_link_pos_w`` for each matched body on every timestep while
    ``current_contact_time > 0`` on the contact sensor. When the body lifts off,
    the buffer keeps the last **on-ground** link pose from the final stance
    frame (not a separate sensor contact point).

    ``has_contact`` is internal episodic state: True once the body has been in
    contact at any time since the last ``reset`` (logical OR over ``in_contact``).
    It is **not** part of :meth:`compute`'s returned tensor.

    Indexing uses articulation ``body_ids`` for link poses and sensor
    ``contact_ids`` for ``current_contact_time``; both resolve the same
    ``body_names`` pattern on their respective bodies list.
    """

    def __init__(self, env, body_names: str, world_frame: bool = False):
        """Args:
            env: Environment instance.
            body_names: Regex or name keys passed to articulation and contact sensor finders.
            world_frame: If True, observation is link position in world frame; if False,
                position relative to the root link (translation subtracted, inverse root quaternion).
        """
        super().__init__(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.world_frame = world_frame
        self.contact_sensor: ContactSensor = self.env.scene.sensors["contact_forces"]
        self.body_ids = self.asset.find_bodies(body_names)[0]

        self.contact_ids = self.contact_sensor.find_bodies(body_names)[0]

        with torch.device(self.device):
            self.body_ids = torch.as_tensor(self.body_ids)
            self.contact_ids = torch.as_tensor(self.contact_ids)
            self.has_contact = torch.zeros(self.num_envs, len(self.body_ids), dtype=bool)
            self.last_contact_pos_w = torch.zeros(self.num_envs, len(self.contact_ids), 3)
        
        if self.env.sim.has_gui() and self.env.backend == "isaac":
            from active_adaptation.envs.backends.isaac import IsaacSceneAdapter
            scene: IsaacSceneAdapter = self.env.scene
            self.marker = scene.create_sphere_marker(
                f"/Visuals/last_contact_pos", color=(0.0, 0.0, 1.0), radius=0.04
            )
        
        self.update()

    def reset(self, env_ids: torch.Tensor) -> None:
        """Clear episodic contact flags and latched positions for ``env_ids``."""
        self.has_contact[env_ids] = False
        self.last_contact_pos_w[env_ids] = 0.0

    def update(self) -> None:
        """Refresh ``in_contact`` from the sensor and slide or freeze ``last_contact_pos_w``."""
        in_contact = self.contact_sensor.data.current_contact_time[:, self.contact_ids] > 0.0
        self.body_link_pos_w = self.asset.data.body_link_pos_w[:, self.body_ids]

        self.has_contact.logical_or_(in_contact)
        self.last_contact_pos_w = torch.where(
            in_contact.unsqueeze(-1),
            self.body_link_pos_w,
            self.last_contact_pos_w
        )

    def compute(self) -> torch.Tensor:
        """Return flattened last-contact link positions (see ``world_frame`` in ``__init__``)."""
        self.root_link_quat_w = self.asset.data.root_link_quat_w
        self.root_link_pos_w = self.asset.data.root_pos_w
        if self.world_frame:
            result = self.last_contact_pos_w
        else:
            result = quat_rotate_inverse(
                self.root_link_quat_w.reshape(self.num_envs, 1, 4),
                self.last_contact_pos_w - self.root_link_pos_w.reshape(self.num_envs, 1, 3)
            )
        return result.reshape(self.num_envs, -1)

    def debug_draw(self) -> None:
        """Draw a vector from each body link to its latched last-contact marker and show spheres."""
        if self.env.sim.has_gui() and self.env.backend == "isaac":
            self.env.debug_draw.vector(
                self.body_link_pos_w,
                self.last_contact_pos_w - self.body_link_pos_w,
                color=(0, 0, 1, 1)
            )
            self.marker.visualize(
                translations=self.last_contact_pos_w.reshape(-1, 3),
            )


class contact_indicator(Observation):
    supported_backends = ("isaac",)
    def __init__(self, env, body_names: str):
        super().__init__(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: ContactSensor = self.env.scene.sensors["contact_forces"]
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
        self.contact_sensor: ContactSensor = self.env.scene.sensors["contact_forces"]
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


# class contact_pos(Observation):
#     """
#     Contact position in world frame.
#     Note that the positions are not contact points, but the positions of the bodies upon contact.
    
#     After reset, both `last_contact_pos_w` and `current_contact_pos_w` are set to 0.0.
#     When in contact,
#         `current_contact_pos_w` is the position where the contact has happened;
#         `last_contact_pos_w` is the position where the last contact has happened.
#     When not in contact,
#         `current_contact_pos_w` is the same as `last_contact_pos_w`, which is the position where the last contact has happened.

#     ``has_contact`` is True for a body once it has touched down in the current episode (latched until reset).
#     ``in_contact`` is True while that body is currently in contact with the scene.

#     Observation vector (per env): ``has_contact``, ``in_contact`` (bool per body each), then ``current_contact_pos``,
#     then ``last_contact_pos`` — each position block is length ``3 * num_bodies`` (world or root frame per ``world_frame``).
#     """

#     def __init__(self, env, body_names: str, world_frame: bool = False):
#         super().__init__(env)
#         self.asset: Articulation = self.env.scene.articulations["robot"]
#         self.world_frame = world_frame
#         self.contact_sensor: ContactSensor = self.env.scene.sensors["contact_forces"]
#         self.body_ids, self.body_names = find_sensor_bodies(
#             self.asset, self.contact_sensor, body_names
#         )

#         if not self.contact_sensor.cfg.track_pose:
#             raise ValueError("The contact sensor must be configured to track pose.")
#         if not self.contact_sensor.cfg.track_air_time:
#             raise ValueError(
#                 "The contact sensor must be configured to track air time "
#                 "(needed for contact / detach events)."
#             )

#         with torch.device(self.device):
#             self.in_contact = torch.zeros(
#                 self.num_envs, len(self.body_ids), dtype=bool
#             )
#             self.has_contact = torch.zeros(
#                 self.num_envs, len(self.body_ids), dtype=bool
#             )
#             self.current_contact_pos_w = torch.zeros(
#                 self.num_envs, len(self.body_ids), 3
#             )
#             self.last_contact_pos_w = torch.zeros(self.num_envs, len(self.body_ids), 3)

#         if self.env.sim.has_gui() and self.env.backend == "isaac":
#             from active_adaptation.envs.backends.isaac import IsaacSceneAdapter

#             scene: IsaacSceneAdapter = self.env.scene
#             self.marker = scene.create_sphere_marker(
#                 f"/Visuals/last_contact_pos", color=(0.0, 0.0, 1.0), radius=0.04
#             )

#     def reset(self, env_ids: torch.Tensor) -> None:
#         self.has_contact[env_ids] = False
#         self.in_contact[env_ids] = False
#         self.current_contact_pos_w[env_ids] = 0.0
#         self.last_contact_pos_w[env_ids] = 0.0

#     def update(self) -> None:
#         data = self.contact_sensor.data
#         first_contact = self.contact_sensor.compute_first_contact(self.env.step_dt)[
#             :, self.body_ids
#         ]
#         first_detached = self.contact_sensor.compute_first_air(self.env.step_dt)[
#             :, self.body_ids
#         ]
#         pos_w = data.pos_w[:, self.body_ids]

#         # Lift-off: in air, current and last both hold this stance's touchdown position.
#         latched = self.current_contact_pos_w[first_detached]
#         self.last_contact_pos_w[first_detached] = latched
#         self.current_contact_pos_w[first_detached] = latched

#         # Touchdown: last ← previous shared value, current ← snapshot at contact.
#         self.last_contact_pos_w[first_contact] = self.current_contact_pos_w[
#             first_contact
#         ]
#         self.current_contact_pos_w[first_contact] = pos_w[first_contact]

#         self.in_contact = data.current_contact_time[:, self.body_ids] > 0.0
#         self.has_contact |= self.in_contact

#     def compute(self) -> torch.Tensor:
#         pos_c = self.current_contact_pos_w
#         pos_l = self.last_contact_pos_w
#         if not self.world_frame:
#             root_pos = self.asset.data.root_link_pos_w.unsqueeze(1)
#             root_quat = self.asset.data.root_link_quat_w.reshape(
#                 self.num_envs, 1, 4
#             )
#             pos_c = quat_rotate_inverse(root_quat, pos_c - root_pos)
#             pos_l = quat_rotate_inverse(root_quat, pos_l - root_pos)
#         obs = torch.cat(
#             [
#                 self.has_contact.reshape(self.num_envs, -1),
#                 self.in_contact.reshape(self.num_envs, -1),
#                 pos_c.reshape(self.num_envs, -1),
#                 pos_l.reshape(self.num_envs, -1),
#             ],
#             dim=-1,
#         )
#         return obs

#     def debug_draw(self) -> None:
#         if self.env.backend == "isaac":
#             self.marker.visualize(
#                 translations=self.last_contact_pos_w.reshape(-1, 3),
#             )

