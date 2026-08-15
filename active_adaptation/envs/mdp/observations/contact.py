import torch
from typing import TYPE_CHECKING
from typing_extensions import override
from .base import ObservationV2
from active_adaptation.utils.math import quat_rotate_inverse
from active_adaptation.utils.symmetry import cartesian_space_symmetry
from tensordict import TensorDictBase

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.sensors import ContactSensor
    from active_adaptation.envs.env_base import _EnvBase


class last_contact_pos(ObservationV2):
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

    def __init__(self, body_names: str, world_frame: bool = False):
        super().__init__()
        self.body_names_pattern = body_names
        self.world_frame = world_frame

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: ContactSensor = self.env.scene.sensors["contact_forces"]
        self.body_ids = self.asset.find_bodies(self.body_names_pattern)[0]
        self.contact_ids = self.contact_sensor.find_bodies(self.body_names_pattern)[0]

        with torch.device(self.device):
            self.body_ids = torch.as_tensor(self.body_ids)
            self.contact_ids = torch.as_tensor(self.contact_ids)
            self.has_contact = torch.zeros(self.num_envs, len(self.body_ids), dtype=bool)
            self.last_contact_pos_w = torch.zeros(self.num_envs, len(self.contact_ids), 3)

        if self.env.sim.has_gui() and self.env.backend == "isaac":
            from active_adaptation.envs.backends.isaac import IsaacSceneAdapter

            scene: IsaacSceneAdapter = self.env.scene
            self.marker = scene.create_sphere_marker(
                "/Visuals/last_contact_pos", color=(0.0, 0.0, 1.0), radius=0.04
            )

        self.update()

    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase) -> None:
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
            self.last_contact_pos_w,
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
                self.last_contact_pos_w - self.root_link_pos_w.reshape(self.num_envs, 1, 3),
            )
        return result.reshape(self.num_envs, -1)

    def debug_draw(self) -> None:
        """Draw a vector from each body link to its latched last-contact marker and show spheres."""
        if self.env.sim.has_gui() and self.env.backend == "isaac":
            self.env.scene.draw_vector(
                self.body_link_pos_w,
                self.last_contact_pos_w - self.body_link_pos_w,
                color=(0, 0, 1, 1),
            )
            self.marker.visualize(
                translations=self.last_contact_pos_w.reshape(-1, 3),
            )


class contact_indicator(ObservationV2):
    supported_backends = ("isaac",)

    def __init__(self, body_names: str):
        super().__init__()
        self.body_names_pattern = body_names

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: ContactSensor = self.env.scene.sensors["contact_forces"]
        self.body_names = self.asset.find_bodies(self.body_names_pattern)[1]
        self.body_ids = self.contact_sensor.find_bodies(self.body_names_pattern)[0]

    def compute(self):
        return self.contact_sensor.data.current_contact_time[:, self.body_ids] > 0.0

    def symmetry_transform(self):
        return cartesian_space_symmetry(self.asset, self.body_names, sign=(1,))


class contact_forces(ObservationV2):
    """Net contact force on matched bodies, flattened to ``(num_envs, 3 * B)``.

    Reads ``force_matrix_w`` when the sensor has a filter (e.g. object vs
    terrain), otherwise ``net_forces_w`` (Isaac) or ``force`` (mjlab).

    Args:
        body_names: Body-name regex on ``entity_name``.
        world_frame: If False (default), rotate forces into the entity root frame.
        entity_name: Scene entity key (``robot``, ``object``, …).
        sensor_name: Scene sensor key (``contact_forces``, ``object_contact``, …).
        vis_scale: Arrow length in meters per Newton for debug draw.
    """

    def __init__(
        self,
        body_names: str,
        world_frame: bool = False,
        entity_name: str = "robot",
        sensor_name: str = "contact_forces",
        vis_scale: float = 0.01,
    ):
        super().__init__()
        self.body_names_pattern = body_names
        self.world_frame = world_frame
        self.entity_name = entity_name
        self.sensor_name = sensor_name
        self.vis_scale = vis_scale

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        self.asset: Articulation | RigidObject = self.env.scene.entities[self.entity_name]
        self.contact_sensor: ContactSensor = self.env.scene.sensors[self.sensor_name]
        self.asset_body_ids, self.body_names = self.asset.find_bodies(self.body_names_pattern)
        if hasattr(self.contact_sensor, "find_bodies"):
            self.sensor_body_ids = self.contact_sensor.find_bodies(
                self.body_names, preserve_order=True
            )[0]
        else:
            names = self.contact_sensor.primary_names
            self.sensor_body_ids = [names.index(name) for name in self.body_names]

    def compute(self):
        self.body_pos_w = self.asset.data.root_link_pos_w
        if self.env.backend == "isaac":
            self.forces_w = self.contact_sensor.data.net_forces_w[:, self.sensor_body_ids]
        elif self.env.backend == "mjlab":
            self.forces_w = self.contact_sensor.data.force[:, self.sensor_body_ids]
        else:
            raise ValueError(f"Unsupported backend: {self.env.backend}")
        contact_forces = self.forces_w
        if not self.world_frame:
            contact_forces = quat_rotate_inverse(
                self.asset.data.root_link_quat_w.reshape(self.num_envs, 1, 4),
                contact_forces,
            )
        return contact_forces.reshape(self.num_envs, -1)

    def symmetry_transform(self):
        if getattr(self.asset.cfg, "spatial_symmetry_mapping", None) is None:
            raise NotImplementedError(
                f"contact_forces symmetry is undefined for entity {self.entity_name!r}"
            )
        return cartesian_space_symmetry(self.asset, self.body_names)

    def debug_draw(self):
        self.env.scene.draw_vector(
            self.body_pos_w.reshape(-1, 3),
            self.forces_w.reshape(-1, 3),
            color=(1.0, 0.25, 0.1, 1.0),
        )
